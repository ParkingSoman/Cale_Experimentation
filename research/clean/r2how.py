"""R2 "how": can a cheap rule rebuild a trained coarse parent from its verified children, with no training run?

Families = trained small-layer units (best pole) that passed the coverage test, with their children (old neuron
poles), hub children removed. For each family, every rule from m1.py (H0-H6) builds ONE parent neuron from the
children only. The target is the children's summed output, exactly as in M1. Then the rule's parent is compared
with the TRAINED parent, which the rules never see:
  act_corr     correlation of activations on held-out text (the trained parent's active pole)
  top_jaccard  overlap of the top-100 held-out tokens
  child_cover  mean percentile of the rule parent's activation on each child's top-50 tokens (1 = fires there)
  out_cos      cosine of the output directions
Diagnostic upper bound "SPAN": the best parent whose gate and value are mixes of the children's gates and values,
fitted to the trained parent's own activation (it sees the trained parent, so it is not a rule).

  python3 research/clean/r2how.py --layer 28 --K 48 --tag global
Writes results/clean-r2how-L{L}-K{K}-{tag}.json.
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "research", "tree"))
sys.path.insert(0, HERE)
from tree_lib import load_model, token_blocks, Weights, hidden, SEQ, silu  # noqa: E402
from r1 import capture, load_w  # noqa: E402
from m1 import run_rules, RULES  # noqa: E402

HUB = 0.20
N_KIDS = 6


def pct(ref_sorted, vals):
    return np.searchsorted(ref_sorted, vals) / len(ref_sorted)


def span_fit(X, a_target, Gk, Uk):
    """Gate and value as mixes of the children's gates and values, fitted by least squares to the target.
    Gate: fit X g ~ the trained parent's gate input is unknown, so alternate: value given gate, gate given value."""
    gs, us = Gk.T, Uk.T                                  # 576 x m
    Xg, Xu = X @ gs, X @ us                              # tokens x m
    g = Gk.mean(0)
    for _ in range(5):
        sg = silu(X @ g)
        A = Xu * sg[:, None]
        b, *_ = np.linalg.lstsq(A, a_target, rcond=None)
        u = us @ b
        v = X @ u
        # Gate step: one linearized least squares on silu(Xg c) * v ~ a_target around the current gate.
        p = X @ g
        s = 1 / (1 + np.exp(-p))
        ds = s * (1 + p * (1 - s))
        J = Xg * (ds * v)[:, None]
        r = a_target - silu(p) * v
        dc, *_ = np.linalg.lstsq(J, r, rcond=None)
        c0, *_ = np.linalg.lstsq(gs, g, rcond=None)
        g = gs @ (c0 + 0.5 * dc)
    return g[None, :], u[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--tag", default="global")
    ap.add_argument("--axis_file", default="clean-r1cover.json")
    args = ap.parse_args()
    L, K, tag = args.layer, args.K, args.tag
    t0 = time.time()
    model, tok = load_model()
    W = Weights(model)
    cal = token_blocks(tok, "train", 64)
    held = token_blocks(tok, "test", 100)[:40]
    XC, XH = capture(model, cal, [L])[L], capture(model, held, [L])[L]
    Xc = XC[np.arange(XC.shape[0]) % SEQ != 0].astype(np.float64)
    Xh = XH[np.arange(XH.shape[0]) % SEQ != 0].astype(np.float64)
    G, U, D = [a.astype(np.float64) for a in (W.G[L], W.U[L], W.D[L])]
    n_old = G.shape[0]
    P = load_w(L, K, tag)
    Hh_old = hidden(Xh, G, U)

    row = next(r for r in json.load(open(os.path.join(ROOT, "results", args.axis_file)))["rows"]
               if r["layer"] == L and r["K"] == K and r["tag"] == tag)
    cnt = collections.Counter(c for u in row["units"] for c in u["children"])
    hubs = {c for c, n in cnt.items() if n / len(row["units"]) > HUB}
    best = {}
    for u in row["units"]:
        kids = [c for c in u["children"] if c not in hubs]
        if len(kids) >= 2 and u["up_cover"] >= 0.5:
            if u["unit"] not in best or len(kids) > len(best[u["unit"]][1]):
                best[u["unit"]] = (u["pole"], kids[:N_KIDS])
    print(f"L{L} K={K} {tag}: {len(best)} families, hubs {len(hubs)} [{time.time() - t0:.0f}s]", flush=True)

    recs = []
    for t, (pole, kids) in sorted(best.items()):
        s_T = 1 if pole == "+" else -1
        idx = np.array([c % n_old for c in kids])
        sg = np.array([1 if c < n_old else -1 for c in kids], dtype=np.float64)
        # Children as neurons whose active direction is positive: flip value and output together for a "-" pole.
        Gc, Uc, Dc = G[idx], U[idx] * sg[:, None], (D[:, idx] * sg[None, :]).T
        a_T = s_T * hidden(Xh, P[0][[t]], P[1][[t]])[:, 0]
        ref_T = np.sort(a_T)
        topT = set(np.argsort(-a_T)[:100].tolist())
        kid_tops = [np.argsort(-(sg[i] * Hh_old[:, idx[i]]))[:50] for i in range(len(idx))]
        d_T = s_T * P[2][:, t]

        def score(Gp, Up, Dp):
            a = hidden(Xh, Gp, Up)[:, 0]
            if np.corrcoef(a, a_T)[0, 1] < 0:            # a single neuron's pole is free: compare the matching pole
                a, Dp = -a, -Dp
            ref = np.sort(a)
            return {"act_corr": float(np.corrcoef(a, a_T)[0, 1]),
                    "top_jaccard": len(set(np.argsort(-a)[:100].tolist()) & topT) / len(set(np.argsort(-a)[:100].tolist()) | topT),
                    "child_cover": float(np.mean([pct(ref, a[k]).mean() for k in kid_tops])),
                    "out_cos": float(Dp.reshape(-1) @ d_T / (np.linalg.norm(Dp) * np.linalg.norm(d_T) + 1e-12))}

        out = run_rules(Xc, Xh, Gc, Uc, Dc, 1)
        rec = {"unit": int(t), "pole": pole, "children": [int(c) for c in kids], "rules": {}}
        for k, (Pp, secs) in out.items():
            rec["rules"][k] = {**score(Pp[0], Pp[1], Pp[2]), "secs": secs}
        # Trained parent's own child coverage, as the reference value.
        rec["trained_child_cover"] = float(np.mean([pct(ref_T, a_T[k]).mean() for k in kid_tops]))
        gS, uS = span_fit(Xh, a_T, Gc, Uc)
        rec["rules"]["SPAN"] = score(gS, uS, d_T[None, :])
        recs.append(rec)
        print(f"T{t:2d}{pole} kids {len(kids)} | " + " ".join(
            f"{k}:{v['act_corr']:+.2f}" for k, v in rec["rules"].items()) + f" [{time.time() - t0:.0f}s]", flush=True)

    names = RULES + ["SPAN"]
    summ = {k: {m: float(np.median([r["rules"][k][m] for r in recs if k in r["rules"]]))
                for m in ("act_corr", "top_jaccard", "child_cover", "out_cos")} for k in names}
    summ["trained_child_cover"] = float(np.median([r["trained_child_cover"] for r in recs]))
    print("\nMEDIAN over families (rule parent vs the TRAINED parent)")
    print(f"{'rule':6s} {'act_corr':>9s} {'top_jacc':>9s} {'child_cov':>10s} {'out_cos':>8s}")
    for k in names:
        s = summ[k]
        print(f"{k:6s} {s['act_corr']:9.2f} {s['top_jaccard']:9.2f} {s['child_cover']:10.2f} {s['out_cos']:8.2f}")
    print(f"trained parent's own child cover: {summ['trained_child_cover']:.2f}")
    json.dump({"layer": L, "K": K, "tag": tag, "summary": summ, "families": recs},
              open(os.path.join(ROOT, "results", f"clean-r2how-L{L}-K{K}-{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
