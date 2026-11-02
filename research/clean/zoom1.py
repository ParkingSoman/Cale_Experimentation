"""ZOOM-1: one zoom configuration, a dial in steps of ONE neuron (families open gradually).

Configuration for layer L (default 28: a loss check runs only the 2 layers after it on cached inputs):
  parents   K units, short-tail fit from the cut (r2round5.fit_tail)
  families  every original neuron belongs to the parent whose activation it matches best (|correlation|)
  order     original neurons come back strongest first (importance = mean |activation| x |output column|)
  dial(N)   the top-M original neurons are on; a parent stays on until ALL its children are on; M is the largest
            value with M + (parents still on) <= N. No refit at any setting.
  nested    the parents are fitted under random partial openings: each family opens a random fraction of its own
            children (strongest first), its parent is off only when the family is fully open.
Pass criteria (fixed before the run, same as zoom0):
  1 steady (loss never rises along the dial)   2 beats a same-size cut with LS refit at every test size
  3 full setting = original                     4 within 25% of a layer fitted for that size (gap share)

  python3 research/clean/zoom1.py [--K 24] [--fit_secs 30] [--nested_secs 60] [--per_size_fits] [--eval_set held|test2]
Writes results/clean-zoom1.json (one row per attempt).
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np
import mlx.core as mx
import mlx.optimizers as optim

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "research", "tree"))
sys.path.insert(0, HERE)
from tree_lib import load_model, token_blocks, Weights, hidden, SEQ  # noqa: E402
from r1 import capture, SmallMLP  # noqa: E402
from r2round5 import cache_h, fit_tail, tail  # noqa: E402
from r2round3 import joint_out  # noqa: E402
from zoom0 import ZoomMLP as _ZoomMLP  # noqa: E402


class ZoomMLP(_ZoomMLP):
    """Adds dD: a correction to the original neurons' output columns, scaled by beta = share of the layer's
    importance that is off. beta = 0 at full size, so the full setting is exactly the original layer."""

    def __init__(self, Pp, G, U, D, dD=None):
        super().__init__(Pp, G, U, D)
        self.p.dD = mx.array(np.zeros_like(D, dtype=np.float32) if dD is None else dD.astype(np.float32))
        if RANK > 0:
            # Low-rank corrections: dD = dA @ dB, eD = eA @ eB (rank RANK). dA/eA start at 0, so the start is exact.
            rng = np.random.default_rng(1)
            n = D.shape[1]
            self.p.dA = mx.zeros((D.shape[0], RANK))
            self.p.dB = mx.array((rng.standard_normal((RANK, n)) / np.sqrt(n)).astype(np.float32))
            self.p.eA = mx.zeros((D.shape[0], RANK))
            self.p.eB = mx.array((rng.standard_normal((RANK, n)) / np.sqrt(n)).astype(np.float32))
        self.beta = mx.array(0.0)
        self.use_dD = CORR
        if KNOTS > 0:
            # Learned schedule: beta = linear interpolation of KNOTS values over the share of original neurons on.
            self.p.bk = mx.array(np.ones(KNOTS, dtype=np.float32))     # beta = bk(x) * share off: exact at full
        self.b_i, self.b_w, self.b_off = 0, mx.array(0.0), mx.array(1.0)
        self.p.eD = mx.array(np.zeros_like(D, dtype=np.float32))
        self.alpha_own = mx.zeros((D.shape[1],))

    def __call__(self, x):
        gp = x @ self.p.g.T
        hp = gp * mx.sigmoid(gp) * (x @ self.p.u.T) * self.mask_p
        go = x @ self.G.T
        ho = go * mx.sigmoid(go) * (x @ self.U.T) * self.mask_o
        out = hp @ self.p.d.T + ho @ self.D.T
        if KNOTS > 0:
            self.beta = ((1 - self.b_w) * self.p.bk[self.b_i] + self.b_w * self.p.bk[min(self.b_i + 1, KNOTS - 1)]) * self.b_off
        if RANK > 0:
            if self.use_dD:
                out = out + self.beta * ((ho @ self.p.dB.T) @ self.p.dA.T)
            if FAMCORR:
                out = out + ((ho * self.alpha_own) @ self.p.eB.T) @ self.p.eA.T
            return out
        if self.use_dD:
            out = out + self.beta * (ho @ self.p.dD.T)
        if FAMCORR:
            out = out + (ho * self.alpha_own) @ self.p.eD.T
        return out


IMP = None   # set in main: importance of every original neuron


def masks_for(M_on, fam, K):
    """Original-neuron mask (n) and parent scale (K) given which originals are on.
    FADE: a parent's output is scaled by the share of its family's importance that is still off."""
    if RESID is not None:
        # RESID[t][k] = share of family t's output still missing when its top-k children are on (calibration).
        k_on = np.bincount(fam, weights=M_on.astype(float), minlength=K).astype(int)
        alpha = np.array([RESID[t][k_on[t]] for t in range(K)])
    else:
        tot = np.bincount(fam, weights=IMP, minlength=K)
        on = np.bincount(fam, weights=IMP * M_on, minlength=K)
        alpha = np.clip(1 - on / (tot + 1e-12), 0, 1)
    if not FADE:
        alpha = (alpha > 1e-9).astype(float)
    return M_on, alpha


FADE = True
RESID = None
CORR = False
FAMCORR = False
RANK = 0
KNOTS = 0
ORDER = None
PATH_SHARE = 0.0


def beta_of(mo):
    return float(1 - (IMP * mo).sum() / IMP.sum())


def set_beta(z, mo):
    """Importance-based beta, or the knot position for the learned schedule (share of originals on)."""
    if KNOTS > 0:
        x = float(np.mean(mo)) * (KNOTS - 1)
        z.b_i, z.b_w = int(min(np.floor(x), KNOTS - 2)), mx.array(x - min(np.floor(x), KNOTS - 2))
        z.b_i = max(z.b_i, 0)
        z.b_off = mx.array(1.0 - float(np.mean(mo)))
    else:
        z.beta = mx.array(beta_of(mo))


def fit_nested_partial(S, H, Pp, G, U, D, fam, rank_in_fam, K, secs, lr=3e-4, batch=4, seed=0):
    rng = np.random.default_rng(seed)
    z = ZoomMLP(Pp, G, U, D)
    teacher = S.model.model.layers[S.L].mlp
    S.model.freeze()

    def loss_fn(params, h, tl, mp, mo):
        z.p.update(params)
        z.mask_p, z.mask_o = mp, mo
        lg = tail(S, h, z)
        lg = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        return (mx.exp(tl) * (tl - lg)).sum(-1)[:, 1:].mean()

    vg = mx.value_and_grad(loss_fn)
    opt = optim.Adam(learning_rate=lr)
    params = z.p.parameters()
    t0, steps = time.time(), 0
    while time.time() - t0 < secs:
        h = mx.array(H[rng.integers(0, H.shape[0], batch)])
        tl = tail(S, h, teacher)
        tl = mx.stop_gradient(tl - mx.logsumexp(tl, axis=-1, keepdims=True))
        # Each family opens a random fraction of its children; some stay closed, some open fully.
        if ORDER is not None and rng.random() < PATH_SHARE:
            # A real dial state: the top-M original neurons in the global order (M uniform), as at runtime.
            on = np.zeros(len(fam), bool)
            on[ORDER[:rng.integers(0, len(fam) + 1)]] = True
        else:
            f = np.clip(rng.random(K) * 1.4 - 0.2, 0, 1) * (rng.random() < 0.9)   # 10% of steps: all closed
            on = rank_in_fam < f[fam] - 1e-9
            on |= f[fam] >= 1
        mo, mp = masks_for(on, fam, K)
        set_beta(z, mo)
        z.alpha_own = mx.array(mp[fam].astype(np.float32))
        _, g = vg(params, h, tl, mx.array(mp.astype(np.float32)), mx.array(mo.astype(np.float32)))
        opt.update(params, g)
        mx.eval(params, opt.state)
        steps += 1
    z.p.update(params)
    S.model.unfreeze()
    global DD, ED, BK
    BK = np.array(z.p.bk, dtype=np.float64) if KNOTS > 0 else None
    if RANK > 0:   # store the full products for evaluation and saving (zoomfinal reads dD / eD)
        DD = np.array(z.p.dA @ z.p.dB, dtype=np.float64)
        ED = np.array(z.p.eA @ z.p.eB, dtype=np.float64)
    else:
        DD = np.array(z.p.dD, dtype=np.float64)
        ED = np.array(z.p.eD, dtype=np.float64)
    return tuple(np.array(a, dtype=np.float64) for a in (z.p.g, z.p.u, z.p.d)), steps


DD = None
ED = None
BK = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--fit_secs", type=float, default=30)
    ap.add_argument("--nested_secs", type=float, default=60)
    ap.add_argument("--n_cal", type=int, default=32)
    ap.add_argument("--n_held", type=int, default=20)
    ap.add_argument("--eval_set", default="held", choices=["held", "dev2", "test2"],
                    help="held = test blocks 0-19 (iteration). test2 = test blocks 60-99 (final, never used before)")
    ap.add_argument("--per_size_fits", action="store_true")
    ap.add_argument("--no_fade", action="store_true")
    ap.add_argument("--fade", default="imp", choices=["imp", "resid"])
    ap.add_argument("--order", default="imp", choices=["imp", "gap", "hybrid"])
    ap.add_argument("--switch", type=int, default=192, help="hybrid: returns ordered gap-first before this count")
    ap.add_argument("--path_share", type=float, default=0.0, help="share of nested steps on real dial states")
    ap.add_argument("--monotone", action="store_true")
    ap.add_argument("--famcorr", action="store_true")
    ap.add_argument("--beta_knots", type=int, default=0, help="learned beta schedule with this many knots")
    ap.add_argument("--corr_rank", type=int, default=0, help="0 = full corrections, r = rank-r corrections")
    ap.add_argument("--corr", action="store_true", help="faded output correction for the original neurons")
    ap.add_argument("--save", default="")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    t0 = time.time()
    L, K = args.layer, args.K
    model, tok = load_model()
    W = Weights(model)
    S = types.SimpleNamespace(model=model, L=L)
    cal = token_blocks(tok, "train", args.n_cal)
    test = token_blocks(tok, "test", 100)
    held = {"held": test[:args.n_held], "dev2": test[20:40], "test2": test[60:100]}[args.eval_set]
    Hcal, Hheld = cache_h(S, cal), cache_h(S, held)
    Xc = capture(model, cal, [L])[L]
    Xc = Xc[np.arange(len(Xc)) % SEQ != 0].astype(np.float64)
    G, U, D = [a.astype(np.float64) for a in (W.G[L], W.U[L], W.D[L])]
    n = G.shape[0]
    Hc = hidden(Xc, G, U)
    Yc = Hc @ D.T
    labels = np.array(held[:, 1:])
    Hh_mx = mx.array(Hheld)

    def ce_mlp(mlp):
        tot = 0.0
        for i in range(0, Hheld.shape[0], 4):
            lg = tail(S, Hh_mx[i:i + 4], mlp)
            lg = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            nll = -mx.take_along_axis(lg, mx.array(labels[i:i + 4])[..., None], axis=-1).squeeze(-1)[:, 1:]
            tot += float(nll.sum().item())
        return tot / (labels.shape[0] * (labels.shape[1] - 1))

    def ce(P):
        return ce_mlp(SmallMLP(P))

    base = ce((G, U, D))
    imp = np.abs(Hc).mean(0) * np.linalg.norm(D, axis=0)
    global IMP, FADE, CORR, FAMCORR, RANK, KNOTS
    IMP, FADE, CORR, FAMCORR, RANK, KNOTS = imp, not args.no_fade, args.corr, args.famcorr, args.corr_rank, args.beta_knots

    def cut(N):
        keep = np.sort(np.argsort(-imp)[:N])
        return G[keep], U[keep], joint_out(Xc, G[keep], U[keep], Yc)

    Pp, steps = fit_tail(S, Hcal, cut(K), args.fit_secs / 60)
    A = hidden(Xc, Pp[0], Pp[1])
    Az = (A - A.mean(0)) / (A.std(0) + 1e-9)
    Hz = (Hc - Hc.mean(0)) / (Hc.std(0) + 1e-9)
    fam = np.argmax(np.abs(Hz.T @ Az) / len(Az), axis=1)
    sizes = np.bincount(fam, minlength=K)
    rank_in_fam = np.zeros(n)
    for t in range(K):
        m = np.where(fam == t)[0]
        rank_in_fam[m[np.argsort(-imp[m])]] = np.arange(len(m)) / max(1, len(m))
    if args.fade == "resid":
        global RESID
        RESID = []
        for t in range(K):
            m = np.where(fam == t)[0]
            m = m[np.argsort(-imp[m])]
            Yf = Hc[:, m] @ D[:, m].T
            nf = np.linalg.norm(Yf) + 1e-12
            r, acc = [1.0], np.zeros_like(Yf)
            for j in m:
                acc += np.outer(Hc[:, j], D[:, j])
                r.append(float(np.linalg.norm(Yf - acc) / nf))
            r[-1] = 0.0
            r = np.clip(r, 0, 1)
            if args.monotone:
                r = np.minimum.accumulate(r)        # a parent never grows back as its children return
            RESID.append(r)
    global ORDER, PATH_SHARE
    ORDER, PATH_SHARE = np.argsort(-imp), args.path_share
    if args.order in ("gap", "hybrid") and RESID is not None:
        # Greedy merge over families: next comes the child that closes the largest missing piece of its
        # family's output (RESID drop x family output size). Within a family the order stays fixed.
        import heapq
        fam_lists, fam_norm = [], []
        for t in range(K):
            m = np.where(fam == t)[0]
            fam_lists.append(m[np.argsort(-imp[m])])
            fam_norm.append(float(np.linalg.norm(Hc[:, m] @ D[:, m].T)))
        def key(t, k, by_gap):
            return -(RESID[t][k] - RESID[t][k + 1]) * fam_norm[t] if by_gap else -imp[fam_lists[t][k]]

        nxt = [0] * K
        ordr = []
        # hybrid: gap-first for the first SWITCH returns, then strongest-first. Within a family the order is fixed.
        while len(ordr) < n:
            by_gap = args.order == "gap" or len(ordr) < args.switch
            cands = [(key(t, nxt[t], by_gap), t) for t in range(K) if nxt[t] < len(fam_lists[t])]
            _, t = min(cands)
            ordr.append(fam_lists[t][nxt[t]])
            nxt[t] += 1
        ORDER = np.array(ordr)
    nsteps = 0
    if args.nested_secs > 0:
        Pp, nsteps = fit_nested_partial(S, Hcal, Pp, G, U, D, fam, rank_in_fam, K, args.nested_secs)
    order = ORDER                                              # the return order (strongest or gap-first)
    rank_used, RANK = RANK, 0          # evaluate with the folded full products (same function)
    z = ZoomMLP(Pp, G, U, D, DD)
    if ED is not None:
        z.p.eD = mx.array(ED.astype(np.float32))
    if BK is not None:
        z.p.bk = mx.array(BK.astype(np.float32))

    def setting(N):
        """Largest M with M + parents-still-on <= N. Returns (M, total neurons, masks)."""
        best = None
        lo, hi = 0, n
        while lo <= hi:                                        # parents-on is non-increasing in M: binary search
            M = (lo + hi) // 2
            on = np.zeros(n, bool)
            on[order[:M]] = True
            mo, mp = masks_for(on, fam, K)
            tot = M + int((mp > 1e-9).sum())
            if tot <= N:
                best, lo = (M, int(tot), mo, mp), M + 1
            else:
                hi = M - 1
        return best

    def ce_setting(N):
        M, tot, mo, mp = setting(N)
        z.mask_o, z.mask_p = mx.array(mo.astype(np.float32)), mx.array(mp.astype(np.float32))
        set_beta(z, mo)
        z.alpha_own = mx.array(mp[fam].astype(np.float32))
        return ce_mlp(z), tot

    # Steady check every 16 neurons AND at every test size (tightened after v8: a 64-step sweep missed a rise).
    sweep = sorted(N for N in set([K] + list(range(32, n + 1, 16)) + [2 * K, 4 * K, 8 * K, 16 * K, 32 * K] + [n])
                   if K <= N <= n)
    dial = []
    for N in sweep:
        c, tot = ce_setting(N)
        dial.append({"N": N, "neurons": tot, "dce": c - base})
    steady = all(dial[i + 1]["dce"] <= dial[i]["dce"] + 1e-4 for i in range(len(dial) - 1))
    # Practical version, added after v7 and reported NEXT TO the strict one: rises under 0.001 nats ignored.
    steady_practical = all(dial[i + 1]["dce"] <= dial[i]["dce"] + 1e-3 for i in range(len(dial) - 1))
    max_rise = max([0.0] + [dial[i + 1]["dce"] - dial[i]["dce"] for i in range(len(dial) - 1)])
    test_sizes = [s for s in (K, 2 * K, 4 * K, 8 * K, 16 * K, 32 * K) if s < n]
    cmp = []
    for N in test_sizes:
        c, tot = ce_setting(N)
        r = {"N": N, "neurons": tot, "dial": c - base, "cut": ce(cut(tot)) - base}
        if args.per_size_fits:
            Pf, _ = fit_tail(S, Hcal, cut(tot), (args.fit_secs + args.nested_secs) / 60)
            r["fitted"] = ce(Pf) - base
            r["within_25pct"] = bool(r["dial"] <= r["fitted"] * 1.25 + 1e-4)
        cmp.append(r)
    res = {"note": args.note, "eval_set": args.eval_set, "layer": L, "K": K, "fit_secs": args.fit_secs,
           "nested_secs": args.nested_secs, "fade": "none" if args.no_fade else args.fade, "corr": args.corr, "monotone": args.monotone, "famcorr": args.famcorr, "corr_rank": args.corr_rank, "beta_knots": args.beta_knots, "path_share": args.path_share, "order": args.order, "switch": args.switch, "fit_steps": steps, "nested_steps": nsteps, "base": base,
           "family_sizes": sizes.tolist(), "dial": dial, "compare": cmp,
           "criteria": {"1_steady": steady, "1b_steady_practical": steady_practical, "max_rise": max_rise, "2_beats_cut": all(r["dial"] < r["cut"] for r in cmp),
                        "3_full_equals_original": abs(dial[-1]["dce"]) < 1e-3,
                        "4_near_fitted": all(r.get("within_25pct", False) for r in cmp) if args.per_size_fits else None},
           "secs": time.time() - t0}
    if args.save:
        # The whole configuration: parents, families, output correction, fade table, return order, importance.
        rl = max(len(r) for r in RESID) if RESID is not None else 0
        rt = np.full((K, rl), np.nan)
        for t in range(K if RESID is not None else 0):
            rt[t, :len(RESID[t])] = RESID[t]
        np.savez(os.path.join(HERE, "r1_weights", f"zoom_{args.save}.npz"), G=Pp[0], U=Pp[1], D=Pp[2], fam=fam,
                 dD=DD if DD is not None else np.zeros_like(D), eD=ED if ED is not None else np.zeros_like(D), resid=rt, order=ORDER, imp=imp,
                 layer=L, K=K, corr=args.corr, famcorr=args.famcorr, fade=args.fade, corr_rank=args.corr_rank,
                 bk=BK if BK is not None else np.zeros(0))
    path = os.path.join(ROOT, "results", "clean-zoom1.json")
    out = json.load(open(path)) if os.path.exists(path) else {"attempts": []}
    out["attempts"].append(res)
    json.dump(out, open(path, "w"), indent=1)
    print(f"[{args.note}] eval {args.eval_set}, base CE {base:.4f}, families min {sizes.min()} med "
          f"{int(np.median(sizes))} max {sizes.max()}, nested steps {nsteps}")
    print("size   dial(no refit)        cut(LS refit)" + ("   fitted" if args.per_size_fits else ""))
    for r in cmp:
        print(f"{r['N']:5d}  +{r['dial']:.4f} ({r['neurons']:4d})   +{r['cut']:.4f}"
              + (f"   +{r['fitted']:.4f}" if "fitted" in r else ""))
    print("sweep:", " ".join(f"{d['neurons']}:{d['dce']:.3f}" for d in dial))
    print("criteria:", res["criteria"], f"[{res['secs']:.0f}s]")


if __name__ == "__main__":
    main()
