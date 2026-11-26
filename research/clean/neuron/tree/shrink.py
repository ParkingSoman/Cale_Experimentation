"""N5 step 2: shrink a capability tree (trees2.json) to a keep fraction f, with no training.

Tree blocks b (in order), members T_b, roots at block l. Kept S_b = top round(f |T_b|) members by value
(std of output x |write|, original network O). Dropped D_b = the rest.
Calibration: 120 group fit photos + 120 other fit photos (not test photos); stage-3 rows use every 2nd position.

Variants (only tree members' weights change, except the two controls):
  drop_goal / drop_all  D's writes removed, bias += D's write x goal-photo mean / all-photo mean
  bias                  sequential: D removed, bias set so the tree part's mean matches O
  rewrite_ind           kept writes re-solved on O's activations, each block on its own (the one-layer method)
  rewrite_seq           kept writes re-solved on the shrunk network C's activations, block by block
  full                  sequential: RE-READ (kept members' input weights re-solved so their pre-activations on C
                        match O's) then RE-WRITE (kept writes re-solved to rebuild the whole tree part of O's output)
  parent                full, and before the sweep the roots' writes also carry the dropped members' output
                        (stage-3 blocks; least squares on O); later targets subtract what the roots now carry
  ctl_absorb            full sweep, but D is absorbed by random NON-tree neurons of the block (same count as S);
                        S keeps its weights
  ctl_tree              full unit applied to a random non-tree "tree" of the same size per block
Depth curve: scores after the first n tree blocks are shrunk, n in DEPTHS. Per-photo KL saved for intervals.
Usage: python shrink.py GROUP [f] [variants...]
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from build_tree import photos, root_scores
from common import DEV, OUT, blocks, coarse_kl_rows, groups, kl_rows, load, net, split

torch.set_grad_enabled(False)
ALL = ["drop_goal", "drop_all", "bias", "rewrite_ind", "rewrite_seq", "full", "parent", "ctl_absorb", "ctl_tree"]


class Stop(Exception):
    pass


def main():
    name = sys.argv[1]
    f = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    variants = sys.argv[3:] or ALL
    T = json.load(open(os.path.join(OUT, "trees2.json")))[name]
    prof = dict(np.load(os.path.join(OUT, "profile_P10.npz")))
    gi = list(prof["groups"]).index(name)
    grp = groups()[name]
    ph = net.Photos("read")
    fit = split(ph.lab)
    m = load("base")
    BL = blocks(m)
    l, roots = T["root_block"], np.array(T["roots"])
    tb = [v["block"] for v in T["levels"]]
    mem = {v["block"]: np.array(v["ids"]) for v in T["levels"]}
    nb = len(tb)
    DEPTHS = sorted(set([1, 2, 4, nb]) & set(range(1, nb + 1)))
    cal_g = photos(ph, grp, fit, 10, max(1, -(-120 // len(grp))), 120)
    others = np.setdiff1d(np.where(fit)[0], np.where(np.isin(ph.lab, grp))[0])
    cal = np.sort(np.concatenate([cal_g, np.random.default_rng(7).choice(others, 120, replace=False)]))
    te_g, te_o = np.array(T["test_group"]), np.array(T["test_other"])
    orig = {b: [p.detach().clone() for p in (BL[b].block[3].weight, BL[b].block[3].bias,
                                               BL[b].block[5].weight, BL[b].block[5].bias)] for b in tb + [l]}
    ls = {b: BL[b].layer_scale.detach().view(-1).cpu().double() for b in tb + [l]}

    def eff(b):  # effective write and bias (layer_scale folded in), float64 cpu
        W1, b1, W2, b2 = orig[b]
        return ls[b][:, None] * W2.cpu().double(), ls[b] * b2.cpu().double()

    def put(b, W1=None, b1=None, Wb=None, bb=None):
        if W1 is not None:
            BL[b].block[3].weight.copy_(W1.float().to(DEV))
            BL[b].block[3].bias.copy_(b1.float().to(DEV))
        if Wb is not None:
            BL[b].block[5].weight.copy_((Wb / ls[b][:, None]).float().to(DEV))
            BL[b].block[5].bias.copy_((bb / ls[b]).float().to(DEV))

    def restore():
        for b in orig:
            put(b, orig[b][0].cpu().double(), orig[b][1].cpu().double(), *eff(b))

    def rows(o, b):  # (B, H, W, C) -> rows, every 2nd position in stage 3
        r = o.reshape(o.shape[0], -1, o.shape[-1])
        if b < 27:
            r = r[:, ::2]
        return r.reshape(-1, o.shape[-1]).cpu().double()

    def capture(b, want_u, extra_root=False):
        """Forward the calibration photos through the CURRENT network up to block b. Returns u (LN output) and g rows."""
        H, out = {}, {"u": [], "g": [], "gr": []}
        hs = [BL[b].block[2].register_forward_hook(lambda mod, i, o: H.__setitem__("u", o)),
              BL[b].block[4].register_forward_hook(lambda mod, i, o: (H.__setitem__("g", o), (_ for _ in ()).throw(Stop())))]
        if extra_root:
            hs.append(BL[l].block[4].register_forward_hook(lambda mod, i, o: H.__setitem__("gr", o)))
        for a in range(0, len(cal), 40):
            try:
                m(ph.tensor(cal[a:a + 40]).to(DEV))
            except Stop:
                pass
            if want_u:
                out["u"].append(rows(H["u"], b))
            out["g"].append(rows(H["g"], b))
            if extra_root:
                out["gr"].append(rows(H["gr"][..., torch.as_tensor(roots, device=DEV)], b))
        for h in hs:
            h.remove()
        return {k: torch.cat(v) if v else None for k, v in out.items()}

    ref = {}
    for tag, ids in (("group", te_g), ("other", te_o)):
        ref[tag] = [m(ph.tensor(ids[a:a + 60]).to(DEV)) for a in range(0, len(ids), 60)]

    def evaluate():
        out = {}
        for tag, ids in (("group", te_g), ("other", te_o)):
            kl, ck = [], []
            for i, a in enumerate(range(0, len(ids), 60)):
                y = m(ph.tensor(ids[a:a + 60]).to(DEV))
                kl.append(kl_rows(ref[tag][i], y).cpu())
                ck.append(coarse_kl_rows(ref[tag][i], y, grp).cpu())
            out[tag] = torch.cat(kl).numpy()
            out[tag + "_coarse"] = torch.cat(ck).numpy()
        return out

    # O pass: per tree block, g rows of O (all hidden), value, kept sets
    t0 = time.time()
    O, UO = {}, {}
    for b in tb:
        c = capture(b, want_u=True)
        O[b] = c["g"].float()
        UO[b] = c["u"].float()
    gr_O = capture(l, want_u=False)["g"][:, roots]  # root rows at block l (stage-3 grid)
    print(f"O pass {time.time() - t0:.0f}s, rows/block {[O[b].shape[0] for b in tb]}", flush=True)
    rng = np.random.default_rng(11)

    def kept(b, ids):
        Wb, _ = eff(b)
        v = O[b][:, ids].double().std(0) * Wb[:, ids].norm(dim=0)
        k = int(round(f * len(ids)))
        order = np.argsort(-v.numpy())
        return ids[order[:k]], ids[order[k:]]

    def solve(X, Y, ridge=1e-6):
        X1 = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1)
        A = X1.T @ X1
        A += ridge * A.diagonal().mean() * torch.eye(len(A), dtype=A.dtype)
        return torch.linalg.solve(A, X1.T @ Y)  # (k+1, d)

    results = {}
    for var in variants:
        restore()
        t1 = time.time()
        tree = dict(mem)
        if var == "ctl_tree":
            tree = {}
            for b in tb:
                conc, br = root_scores(prof, b, grp)
                pool = np.where(~((conc > .4) & (br > .25)))[0]
                tree[b] = rng.choice(pool, len(mem[b]), replace=False)
        parent_A = {}
        if var == "parent":  # roots absorb stage-3 dropped output, solved on O
            Wl, bl = eff(l)
            Wl, bl = Wl.clone(), bl.clone()
            gr = gr_O
            grc = gr - gr.mean(0)
            for b in tb:
                if b >= 27:
                    continue
                S, D = kept(b, tree[b])
                Wb, _ = eff(b)
                dO = O[b][:, D].double() @ Wb[:, D].T
                A = torch.linalg.solve(grc.T @ grc + 1e-6 * torch.eye(len(roots), dtype=grc.dtype) * grc.pow(2).sum(0).mean(),
                                       grc.T @ (dO - dO.mean(0)))  # (roots, d)
                parent_A[b] = A
                Wl[:, roots] += A.T
                bl -= A.T @ gr.mean(0)
            put(l, Wb=Wl, bb=bl)
        cur = {}
        reread_shift = {}
        for n, b in enumerate(tb, 1):
            ids = tree[b]
            S, D = kept(b, ids)
            Wb, bb = eff(b)
            Wnew, bnew = Wb.clone(), bb.clone()
            yT = O[b][:, ids].double() @ Wb[:, ids].T  # tree part of O's output (rows x d)
            if var in ("drop_goal", "drop_all"):
                gm = prof[f"gm{b}"][gi if var == "drop_goal" else -1][D]
                Wnew[:, D] = 0
                bnew = bb + Wb[:, D] @ torch.from_numpy(gm).double()
                put(b, Wb=Wnew, bb=bnew)
            elif var == "rewrite_ind":
                beta = solve(O[b][:, S].double(), yT) if len(S) else solve(torch.zeros(len(yT), 0, dtype=torch.float64), yT)
                Wnew[:, ids] = 0
                Wnew[:, S] = beta[:len(S)].T
                put(b, Wb=Wnew, bb=bb + beta[len(S)])
            else:
                need_u = var in ("full", "parent", "ctl_tree", "ctl_absorb")
                c = capture(b, want_u=need_u, extra_root=(var == "parent" and b < 27))
                W1, b1 = orig[b][0].cpu().double(), orig[b][1].cpu().double()
                gS = c["g"][:, S]
                if need_u and len(S) and var != "ctl_absorb":  # RE-READ: match O's pre-activations of S
                    U = c["u"]
                    P = UO[b].double() @ W1[S].T + b1[S]
                    U1 = torch.cat([U, torch.ones(len(U), 1, dtype=U.dtype)], 1)
                    A = U1.T @ U1
                    lam = 1e-4 * A.diagonal()[:-1].mean()
                    Rg = torch.ones(len(A), dtype=A.dtype)
                    Rg[-1] = 0
                    old = torch.cat([W1[S].T, b1[S][None]], 0)
                    sol = torch.linalg.solve(A + lam * torch.diag(Rg), U1.T @ P + lam * Rg[:, None] * old)
                    W1n, b1n = W1.clone(), b1.clone()
                    W1n[S], b1n[S] = sol[:-1].T, sol[-1]
                    put(b, W1=W1n, b1=b1n)
                    reread_shift[b] = float(((W1n[S] - W1[S]).norm(dim=1) / W1[S].norm(dim=1)).median())
                    gS = F.gelu(U @ W1n[S].T + b1n[S])
                tgt = yT.clone()
                if var == "parent" and b in parent_A:
                    tgt -= (c["gr"] - gr_O.mean(0)) @ parent_A[b]
                if var == "bias":
                    Wnew[:, D] = 0
                    bnew = bb + (tgt - gS @ Wb[:, S].T).mean(0)
                    put(b, Wb=Wnew, bb=bnew)
                elif var == "ctl_absorb":
                    pool = np.setdiff1d(np.arange(Wb.shape[1]), ids)
                    Rn = rng.choice(pool, max(len(S), 1), replace=False)
                    tgt2 = tgt + O[b][:, Rn].double() @ Wb[:, Rn].T - gS @ Wb[:, S].T
                    beta = solve(c["g"][:, Rn], tgt2)
                    Wnew[:, D] = 0
                    Wnew[:, Rn] = beta[:len(Rn)].T
                    put(b, Wb=Wnew, bb=bb + beta[len(Rn)])
                else:
                    beta = solve(gS, tgt)
                    Wnew[:, ids] = 0
                    Wnew[:, S] = beta[:len(S)].T
                    put(b, Wb=Wnew, bb=bb + beta[len(S)])
            if n in DEPTHS:
                cur[n] = evaluate()
                print(f"  {var} depth {n}: KL group {cur[n]['group'].mean():.5f} other {cur[n]['other'].mean():.5f} "
                      f"coarse {cur[n]['group_coarse'].mean():.5f} [{time.time() - t1:.0f}s]", flush=True)
        if reread_shift:
            print(f"  {var} re-read: median relative change of kept reads per block "
                  f"{ {k: round(v, 4) for k, v in reread_shift.items()} }", flush=True)
        results[var] = cur
        if os.environ.get("SAVE_STATE"):
            keep = {b: kept(b, tree[b])[0].tolist() for b in tb}
            torch.save(dict(blocks={b: [p.detach().cpu().clone() for p in (BL[b].block[3].weight, BL[b].block[3].bias,
                                                                          BL[b].block[5].weight, BL[b].block[5].bias)]
                                    for b in orig}, kept=keep, tree={b: np.asarray(tree[b]).tolist() for b in tb},
                            root_block=l, roots=roots.tolist(), cal=cal.tolist()),
                       os.path.join(OUT, f"shrink_state_{name}_{var}_f{f}.pt"))
        np.savez(os.path.join(OUT, f"shrink_{name}_f{f}{os.environ.get('TAG', '')}.npz"),
                 **{f"{v}|{n}|{k}": a for v, d in results.items() for n, e in d.items() for k, a in e.items()})
    restore()


if __name__ == "__main__":
    main()
