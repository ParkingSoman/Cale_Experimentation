"""HOW round 1: build the parent's GATE from data (where the children fire), not from the children's gate weights.

Root cause E1 (CHILDREN-LOG): a SwiGLU gate is one half-space. Mixing the children's gate directions may not reach the
region where ANY child fires. These rules look for a direction that separates that union from the rest:
  H0      keep the dominant child (cut), refit its output             (baseline from M1)
  H6      M1's local gradient fit to the children's summed output     (best M1 rule)
  PROTO   gate = mean input on the union tokens minus the mean input   (prototype)
  PROBE   gate = ridge regression of a +1 / -1 union label on the input (closed-form linear probe, LDA-like)
  LOGIT   gate = logistic regression on the union label, balanced      (convex, seconds)
For the data rules the value and output are then fitted in closed form to the children's summed output.
None of them sees the trained parent.

  python3 research/clean/r2round1.py --layer 28
"""
import argparse
import json
import os
import time

import numpy as np

from r2lib import Setup, score, scale_gate, union_labels, summarize, print_summary, ROOT, silu, relerr
from m1 import run_rules


def probe_ridge(X, y, w, lam=1e-3):
    A = (X * w[:, None]).T @ X
    A += np.eye(A.shape[0]) * lam * np.trace(A) / A.shape[0]
    return np.linalg.solve(A, (X * w[:, None]).T @ y)


def probe_logistic(X, y01, steps=300, lr=0.5):
    """Balanced logistic regression by plain gradient descent on standardized inputs."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    pos = y01.mean()
    w_s = np.where(y01 > 0, 0.5 / pos, 0.5 / (1 - pos))
    beta, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(Z @ beta + b)))
        gr = (p - y01) * w_s
        beta -= lr * (Z.T @ gr / len(y01) + 1e-4 * beta)
        b -= lr * gr.mean()
    # Back to raw inputs. The intercept has no slot (no MLP bias), so it is dropped; near-constant input
    # coordinates can carry part of it (found at mid layers in M1).
    return beta / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--tag", default="global")
    args = ap.parse_args()
    S = Setup(args.layer, args.K, args.tag)
    t0 = time.time()
    rows = []
    for fam in S.families:
        r = {}
        out = run_rules(S.Xc, S.Xh, fam["Gc"], fam["Uc"], fam["Dc"], 1, rules=["H0", "H6"])
        for k, (P, _) in out.items():
            r[k] = score(S, fam, P[0][0], P[1][0], P[2][0])
        y = union_labels(fam)
        Xc = S.Xc
        gdir = {"PROTO": Xc[y].mean(0) - Xc.mean(0),
                "PROBE": probe_ridge(Xc, np.where(y, 1.0, -1.0), np.where(y, 0.5 / y.mean(), 0.5 / (1 - y.mean()))),
                "LOGIT": probe_logistic(Xc, y.astype(np.float64))}
        for k, gd in gdir.items():
            g, u, d = scale_gate(Xc, gd, fam["Y_cal"])
            r[k] = score(S, fam, g, u, d)
        r["unit"] = fam["unit"]
        rows.append(r)
        print(f"T{fam['unit']:2d}{fam['pole']} kids {len(fam['idx'])} | " + " ".join(
            f"{k}:{r[k]['act_corr']:+.2f}/{r[k]['kids_out_err']:.2f}" for k in ("H0", "H6", "PROTO", "PROBE", "LOGIT"))
            + f" [{time.time() - t0:.0f}s]", flush=True)
    names = ["H0", "H6", "PROTO", "PROBE", "LOGIT"]
    summ = summarize(rows, names)
    print_summary(summ, f"ROUND 1, layer {args.layer} ({len(rows)} families). act_corr = match with the trained parent. "
                        "kids_err = error on the children's summed output (lower is better).")
    json.dump({"layer": args.layer, "summary": summ, "rows": rows},
              open(os.path.join(ROOT, "results", f"clean-r2round1-L{args.layer}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
