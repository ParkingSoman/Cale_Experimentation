"""HOW round 2: loss-aware fitting (E4) and a loss-based score (E5).

E4: a coarse parent cannot (and should not) copy the children's summed output in full detail. Fit it under the
    empirical Fisher metric of the layer's output: errors in directions that move the next-word loss count more.
    F = mean over tokens of g g^T, g = d(loss)/d(MLP output at layer L), from the ORIGINAL model. One gradient pass.
E5: swap test. Inside the trained small layer, replace unit t by the rule's parent and measure held-out loss.
    recovered = 1 - (CE_rule - CE_trained) / (CE_zero - CE_trained). 1 = as good as the trained unit, 0 = as bad as
    deleting it.

Rules (none sees the trained parent):
  ZERO, H0, H6            references and the best M1 rule
  H0F, H6F                same gates, value and output refit under the Fisher metric
  PROBEF                  union-probe gate (round 1), Fisher-metric value/output
  python3 research/clean/r2round2.py --layer 28
"""
import argparse
import json
import os
import time

import numpy as np
import mlx.core as mx

from r2lib import Setup, score, union_labels, relerr, ROOT, silu
from r2round1 import probe_ridge
from r1 import SmallMLP
from m1 import run_rules


def fisher(S, n_blocks=16, batch=4):
    """Empirical Fisher of the loss w.r.t. layer L's MLP output, original model, calibration text."""
    L = S.L
    mlp = S.model.model.layers[L].mlp

    class Add:
        def __init__(self, inner):
            self.inner, self.delta = inner, None

        def __call__(self, x):
            return self.inner(x) + self.delta

    add = Add(mlp)
    S.model.model.layers[L].mlp = add
    F = np.zeros((576, 576))
    n = 0
    blocks = S.cal_blocks[:n_blocks]
    for i in range(0, blocks.shape[0], batch):
        b = blocks[i:i + batch]
        x, y = b[:, :-1], b[:, 1:]

        def loss(delta):
            add.delta = delta
            lg = S.model(x)
            lg = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            return -mx.take_along_axis(lg, y[..., None], axis=-1).sum()

        g = mx.grad(loss)(mx.zeros((x.shape[0], x.shape[1], 576)))
        g = np.array(g, dtype=np.float64)[:, 1:, :].reshape(-1, 576)    # drop the first position
        F += g.T @ g
        n += g.shape[0]
    S.model.model.layers[L].mlp = mlp
    return F / n


def fit_value_out_metric(X, g, Y, R, ridge=1e-4):
    """Gate fixed. Fit u, d (rank 1) minimizing sum_t ||(Y_t - a_t d) R^T||^2 with a_t = silu(x g)(x u)."""
    from r2lib import fit_value_out
    Ri = np.linalg.inv(R)
    u, d2 = fit_value_out(X, g, Y @ R.T, ridge)
    return u, d2 @ Ri.T


def ce_blocks(model, blocks, batch=4):
    out = []
    for i in range(0, blocks.shape[0], batch):
        b = blocks[i:i + batch]
        lg = model(b[:, :-1])
        lg = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        out.append(np.array(-mx.take_along_axis(lg, b[:, 1:, None], axis=-1).squeeze(-1).mean(1)))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--tag", default="global")
    ap.add_argument("--eval_blocks", type=int, default=40)
    args = ap.parse_args()
    S = Setup(args.layer, args.K, args.tag)
    t0 = time.time()
    F = fisher(S)
    ev, V = np.linalg.eigh(F)
    ev = np.maximum(ev, ev.max() * 1e-6)
    R = (V * np.sqrt(ev)) @ V.T                           # F = R^T R, symmetric square root
    print(f"Fisher done: top-10 eigen share {ev[-10:].sum() / ev.sum():.2f} [{time.time() - t0:.0f}s]", flush=True)

    from mlx_lm.models.llama import MLP  # noqa: F401  (only to be clear the small layer is a SwiGLU MLP)
    small = SmallMLP(S.P)
    orig = S.model.model.layers[S.L].mlp
    S.model.model.layers[S.L].mlp = small
    held = S.held_blocks[:args.eval_blocks]
    ce_T = ce_blocks(S.model, held)
    Pw = [np.array(small.gate_proj.weight), np.array(small.up_proj.weight), np.array(small.down_proj.weight)]

    def swap(t, g, u, d):
        G2, U2, D2 = [w.copy() for w in Pw]
        G2[t], U2[t], D2[:, t] = g, u, d
        small.gate_proj.weight, small.up_proj.weight = mx.array(G2), mx.array(U2)
        small.down_proj.weight = mx.array(D2)
        ce = ce_blocks(S.model, held)
        small.gate_proj.weight, small.up_proj.weight = mx.array(Pw[0]), mx.array(Pw[1])
        small.down_proj.weight = mx.array(Pw[2])
        return float(ce.mean() - ce_T.mean())

    rows = []
    for fam in S.families:
        t, s = fam["unit"], fam["s_T"]
        r = {"unit": t}
        d_zero = swap(t, Pw[0][t], Pw[1][t], np.zeros(576))
        r["ZERO"] = {"dce": d_zero}
        out = run_rules(S.Xc, S.Xh, fam["Gc"], fam["Uc"], fam["Dc"], 1, rules=["H0", "H6"])
        cands = {k: (P[0][0], P[1][0], P[2][0]) for k, (P, _) in out.items()}
        for k in ("H0", "H6"):
            g = cands[k][0]
            cands[k + "F"] = (g,) + fit_value_out_metric(S.Xc, g, fam["Y_cal"], R)
        y = union_labels(fam)
        gp = probe_ridge(S.Xc, np.where(y, 1.0, -1.0), np.where(y, 0.5 / y.mean(), 0.5 / (1 - y.mean())))
        gp = gp * 2 / (np.std(S.Xc @ gp) + 1e-9)
        cands["PROBEF"] = (gp,) + fit_value_out_metric(S.Xc, gp, fam["Y_cal"], R)
        for k, (g, u, d) in cands.items():
            sc = score(S, fam, g, u, d)
            # The parent's sign: the trained unit's pole is s; the small layer stores the raw unit, so flip u, d.
            dce = swap(t, g, s * u, s * d)
            dce2 = swap(t, g, -s * u, -s * d)          # a rule's pole is free: keep the better sign
            sc["dce"] = min(dce, dce2)
            sc["recovered"] = 1 - sc["dce"] / (d_zero + 1e-9)
            r[k] = sc
        rows.append(r)
        print(f"T{t:2d} zero {d_zero:+.4f} | " + " ".join(f"{k}:{r[k]['recovered']:+.2f}" for k in cands)
              + f" [{time.time() - t0:.0f}s]", flush=True)
    S.model.model.layers[S.L].mlp = orig
    names = ["H0", "H6", "H0F", "H6F", "PROBEF"]
    summ = {k: {m: float(np.median([r[k][m] for r in rows])) for m in ("recovered", "dce", "act_corr", "child_cover",
                                                                       "kids_out_err", "out_cos")} for k in names}
    summ["ZERO_dce_median"] = float(np.median([r["ZERO"]["dce"] for r in rows]))
    print(f"\nROUND 2, layer {args.layer}: median over {len(rows)} families. recovered = share of the trained unit's "
          f"value the rule keeps (swap test). zero-unit dCE median {summ['ZERO_dce_median']:+.4f}")
    print(f"{'rule':8s} {'recovered':>10s} {'act_corr':>9s} {'child_cov':>10s} {'kids_err':>9s} {'out_cos':>8s}")
    for k in names:
        s = summ[k]
        print(f"{k:8s} {s['recovered']:10.2f} {s['act_corr']:9.2f} {s['child_cover']:10.2f} {s['kids_out_err']:9.2f} "
              f"{s['out_cos']:8.2f}")
    json.dump({"layer": args.layer, "summary": summ, "rows": rows},
              open(os.path.join(ROOT, "results", f"clean-r2round2-L{args.layer}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
