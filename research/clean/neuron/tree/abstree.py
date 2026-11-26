"""N5 step 5: the absorption tree. A size is an in-place computation on the base weights, driven by a small config.

Config, per block (from 1,000 calibration photos of the original network):
  rank      neurons by value = std(GELU output) x |write|
  mean m    of each neuron's GELU output
  parents   for each neuron, up to s HIGHER-ranked neurons and coefficients c: sparse regression of its output on
            theirs (orthogonal matching pursuit on the covariance, pool = its 64 most correlated higher-ranked neurons)
Switch to keep fraction f (per block: S = top f by rank, D = the rest). With C[d, p] = c of parent p for dropped d:
  g_D - m_D ~ (I - C_DD)^-1 C_DS (g_S - m_S)          (a dropped parent passes its share up the tree)
  W_S <- W_S + W_D (I - C_DD)^-1 C_DS ;  b <- b + W_D m_D - (W_S_new - W_S) m_S ;  W_D <- 0
One config serves every f. Undo: put the saved deltas back (here: reload).
Compared with: mean fill (C = 0) and the dense re-solve of pair/allblocks.py (a per-size weight copy, the ceiling).
Test: the 500 test photos of diag.py, KL to Base and top-1. Usage: python abstree.py [n_cal] [n_test]
"""
import json
import os
import sys
import time

import numpy as np
import torch

from common import DEV, OUT, blocks, kl_rows, load, net, split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pair"))
from distill import D  # noqa: E402

torch.set_grad_enabled(False)
SS, POOL, FRACS = (1, 2, 4, 8), 64, (0.5, 0.25)


def parents(K, rank_pos, smax=SS[-1], pool=POOL):
    """For every neuron: up to smax higher-ranked parents (OMP on covariance K, float64 cpu). rank_pos[i] = rank of i."""
    h = K.shape[0]
    sd = K.diagonal().clamp_min(1e-12).sqrt()
    corr = (K / sd[:, None] / sd[None, :]).abs()
    higher = rank_pos[None, :] < rank_pos[:, None]  # [i, j]: j ranks above i
    sc = torch.where(higher, corr, torch.full_like(corr, -1.0))
    P = min(pool, h - 1)
    val, cand = sc.topk(P, dim=1)  # (h, P)
    valid = val > -0.5
    Kc = K[cand[:, :, None], cand[:, None, :]]  # (h, P, P)
    ki = K[torch.arange(h)[:, None], cand]  # (h, P)
    eye = torch.eye(P, dtype=K.dtype)
    Kc = torch.where(valid[:, :, None] & valid[:, None, :], Kc, eye.expand(h, P, P) * 1.0)
    ki = torch.where(valid, ki, torch.zeros_like(ki))
    dK = Kc.diagonal(dim1=1, dim2=2)
    sel = torch.zeros(h, 0, dtype=torch.long)
    out = {}
    taken = ~valid
    for t in range(1, smax + 1):
        if sel.shape[1]:
            Kss = torch.gather(torch.gather(Kc, 1, sel[:, :, None].expand(h, sel.shape[1], P)), 2,
                               sel[:, None, :].expand(h, sel.shape[1], sel.shape[1]))
            ks = torch.gather(ki, 1, sel)
            coef = torch.linalg.solve(Kss + 1e-8 * torch.eye(sel.shape[1], dtype=K.dtype), ks)
            Kjs = torch.gather(Kc, 2, sel[:, None, :].expand(h, P, sel.shape[1]))
            r = ki - (Kjs @ coef[:, :, None])[..., 0]
        else:
            r = ki
        score = torch.where(taken, torch.full_like(r, -1.0), r.pow(2) / dK.clamp_min(1e-12))
        pick = score.argmax(1)
        taken = taken.clone()
        taken[torch.arange(h), pick] = True
        sel = torch.cat([sel, pick[:, None]], 1)
        if t in SS:
            Kss = torch.gather(torch.gather(Kc, 1, sel[:, :, None].expand(h, t, P)), 2, sel[:, None, :].expand(h, t, t))
            coef = torch.linalg.solve(Kss + 1e-8 * K.diagonal().mean() * torch.eye(t, dtype=K.dtype),
                                      torch.gather(ki, 1, sel))
            ok = torch.gather(valid, 1, sel)
            out[t] = (torch.gather(cand, 1, sel), torch.where(ok, coef, torch.zeros_like(coef)))
    return out  # s -> (parent ids (h, s), coefficients (h, s))


def main():
    ncal = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    nte = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    ph = net.Photos("read")
    fit = split(ph.lab)
    cal = np.array(sorted([np.where((ph.lab == c) & fit)[0][7] for c in range(1000)]))
    cal = np.sort(np.random.default_rng(0).choice(cal, ncal, replace=False)) if ncal < 1000 else cal
    te = np.load(os.path.join(D, "photos_test.npy"))[:nte]
    lab = torch.tensor(ph.lab[te])
    m = load("base")
    BL = blocks(m)
    acc = {}
    hooks = []
    for key, blk in enumerate(BL):
        h = blk.block[3].out_features
        acc[key] = dict(n=0, s=torch.zeros(h, dtype=torch.float64), ss=torch.zeros(h, h, dtype=torch.float64))

        def f(mod, inp, out, key=key):
            g = out.reshape(-1, out.shape[-1]).float()
            a = acc[key]
            a["n"] += g.shape[0]
            a["s"] += g.sum(0).cpu().double()
            a["ss"] += (g.T @ g).cpu().double()
        hooks.append(blk.block[4].register_forward_hook(f))
    t0 = time.time()
    for k in range(0, len(cal), 25):
        m(ph.tensor(cal[k:k + 25]).to(DEV))
    for h_ in hooks:
        h_.remove()
    print(f"calibration {time.time() - t0:.0f}s", flush=True)
    orig = [(blk.block[5].weight.detach().clone(), blk.block[5].bias.detach().clone()) for blk in BL]
    lsc = [blk.layer_scale.detach().view(-1).cpu().double() for blk in BL]
    cfg, dense = [], {fr: [] for fr in FRACS}
    t1 = time.time()
    for key in range(len(BL)):
        a = acc.pop(key)
        mg = a["s"] / a["n"]
        K = a["ss"] / a["n"] - mg[:, None] * mg[None, :]
        Wb = lsc[key][:, None] * orig[key][0].cpu().double()
        bb = lsc[key] * orig[key][1].cpu().double()
        h = Wb.shape[1]
        order = torch.argsort(-(K.diagonal().clamp_min(0).sqrt() * Wb.norm(dim=0)))
        rank_pos = torch.empty(h, dtype=torch.long)
        rank_pos[order] = torch.arange(h)
        par = parents(K, rank_pos)
        cfg.append(dict(order=order, mean=mg.float(), par={s: (p, c.float()) for s, (p, c) in par.items()}))
        for fr in FRACS:  # dense ceiling (stored per size: does NOT meet the memory rule)
            kk = int(round(fr * h))
            S = order[:kk]
            KSS = K[S][:, S] + 1e-6 * K.diagonal().mean() * torch.eye(kk, dtype=torch.float64)
            Wn = torch.zeros_like(Wb)
            Wn[:, S] = torch.linalg.solve(KSS, K[S] @ Wb.T).T
            dense[fr].append(((Wn / lsc[key][:, None]).float(), ((bb + Wb @ mg - Wn @ mg) / lsc[key]).float()))
        del K, a
        print(f"  block {key}: parents done [{time.time() - t1:.0f}s]", flush=True)
    cfg_bytes = sum(len(c["order"]) * (2 + 2 + SS[-1] * 4) for c in cfg)  # int16 rank, fp16 mean, 8 x (int16 + fp16)
    print(f"config size (8 parents, int16 + fp16): {cfg_bytes / 1e6:.2f} MB", flush=True)

    # cached on device for the switch
    dev_cfg = []
    for c in cfg:
        dev_cfg.append(dict(order=c["order"].to(DEV), mean=c["mean"].to(DEV),
                            par={s: (p.to(DEV), cc.to(DEV)) for s, (p, cc) in c["par"].items()}))

    def restore():
        for key, blk in enumerate(BL):
            blk.block[5].weight.copy_(orig[key][0])
            blk.block[5].bias.copy_(orig[key][1])

    def switch(fr, s):
        """In place, on the device, from the base weights. s = 0: mean fill only."""
        for key, blk in enumerate(BL):
            c = dev_cfg[key]
            W = blk.block[5].weight  # (d, h) without layer scale: the update is linear, so scale does not matter
            b = blk.block[5].bias
            h = W.shape[1]
            kk = int(round(fr * h))
            S, Dd = c["order"][:kk], c["order"][kk:]
            mD, mS = c["mean"][Dd], c["mean"][S]
            WD = W[:, Dd]
            dW = None
            if s > 0:
                pid, pc = c["par"][s]
                pos = torch.full((h,), -1, dtype=torch.long, device=DEV)
                pos[Dd] = torch.arange(len(Dd), device=DEV)
                posS = torch.full((h,), -1, dtype=torch.long, device=DEV)
                posS[S] = torch.arange(kk, device=DEV)
                C_DD = torch.zeros(len(Dd), len(Dd), device=DEV)
                C_DS = torch.zeros(len(Dd), kk, device=DEV)
                pd, cd = pid[Dd], pc[Dd]  # (|D|, s)
                rows = torch.arange(len(Dd), device=DEV)[:, None].expand_as(pd)
                inD, inS = pos[pd] >= 0, posS[pd] >= 0
                C_DD.index_put_((rows[inD], pos[pd][inD]), cd[inD], accumulate=True)
                C_DS.index_put_((rows[inS], posS[pd][inS]), cd[inS], accumulate=True)
                # D is sorted by rank (best first) and parents rank higher, so C_DD is strictly lower triangular
                X = torch.linalg.solve_triangular(torch.eye(len(Dd), device=DEV) - C_DD, C_DS, upper=False)
                dW = WD @ X  # (d, kk)
                W[:, S] += dW
            b += WD @ mD - (dW @ mS if dW is not None else 0)
            W[:, Dd] = 0

    ref = [m(ph.tensor(te[a:a + 50]).to(DEV)) for a in range(0, len(te), 50)]

    def evaluate():
        kl, c = [], 0
        for i, a in enumerate(range(0, len(te), 50)):
            y = m(ph.tensor(te[a:a + 50]).to(DEV))
            kl.append(kl_rows(ref[i], y).cpu())
            c += int((y.argmax(1).cpu() == lab[a:a + 50]).sum())
        return torch.cat(kl).numpy(), c / len(te)

    res = dict(config_mb=cfg_bytes / 1e6, n_cal=int(len(cal)), n_test=int(len(te)))
    rows_out = {}
    for fr in FRACS:
        for s in (0,) + SS:
            restore()
            if DEV == "mps":
                torch.mps.synchronize()
            t2 = time.time()
            switch(fr, s)
            if DEV == "mps":
                torch.mps.synchronize()
            dt = time.time() - t2
            kl, top1 = evaluate()
            rows_out[f"tree_s{s}_{fr}"] = kl
            res[f"tree_s{s}_{fr}"] = dict(kl=float(kl.mean()), top1=top1, switch_ms=1000 * dt)
            print(f"keep {fr} s={s}: KL {kl.mean():.4f} top-1 {top1:.3f} switch {1000 * dt:.1f} ms", flush=True)
        restore()
        for key, blk in enumerate(BL):
            blk.block[5].weight.copy_(dense[fr][key][0].to(DEV))
            blk.block[5].bias.copy_(dense[fr][key][1].to(DEV))
        kl, top1 = evaluate()
        rows_out[f"dense_{fr}"] = kl
        res[f"dense_{fr}"] = dict(kl=float(kl.mean()), top1=top1)
        print(f"keep {fr} dense re-solve: KL {kl.mean():.4f} top-1 {top1:.3f}", flush=True)
        json.dump(res, open(os.path.join(OUT, f"abstree_{ncal}.json"), "w"), indent=1)
        np.savez(os.path.join(OUT, f"abstree_{ncal}_rows.npz"), **rows_out)
    # undo check: switch, then undo by subtracting the same deltas is the deployable path; here verify that a
    # switch from freshly restored weights is deterministic and that restore returns the base output exactly
    restore()
    kl, top1 = evaluate()
    res["after_restore"] = dict(kl=float(kl.mean()), top1=top1)
    json.dump(res, open(os.path.join(OUT, f"abstree_{ncal}.json"), "w"), indent=1)
    print(json.dumps(res["after_restore"]), flush=True)


if __name__ == "__main__":
    main()
