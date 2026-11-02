# serves: T27
"""Check of Theorem T27 part 1 (COARSE-THEORY.md B.Z): the exact rule for fold builds.

Random networks with n = 4 units in d = 6 inputs, two logits; mixes of 7 random points. Groups: {0, 1}, {2, 3} (s = 2).
Compare the exact fold rule (trunk_exact.exact_min_fold) with gradient training of the same fold class (weights
m = |parameters|, 30 random starts), for k = 1 and k = 2, KL and D2.
Prediction (written before the run): training never ends below the exact rule (beyond 1e-6).
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
from trunk_exact import exact_min_fold, per_point_loss  # noqa: E402

rng = np.random.default_rng(3)
viol = 0
rows = []
for trial in range(4):
    d, n, nP = 6, 4, 7
    W = rng.normal(size=(n, d)); b = rng.normal(size=n); a = rng.normal(size=n) * 2
    X = rng.normal(size=(nP, d))
    Z = X @ W.T + b
    tgap = np.maximum(Z, 0) @ a
    pi = np.full(nP, 1 / nP)
    groups = [[0, 1], [2, 3]]
    Zs = [Z[:, gidx] for gidx in groups]
    for k in (1, 2):
        for kind in ("kl", "d2"):
            v, _, _ = exact_min_fold(Zs, pi, tgap, k, kind, starts=2)
            lossf, _ = per_point_loss(kind, tgap)
            best = np.inf
            for _ in range(30):
                gsel = rng.integers(0, 2, size=k)

                def F(p):
                    out = p[-1] * np.ones(nP)
                    for j in range(k):
                        u = np.abs(p[3 * j:3 * j + 3])
                        zeta = np.hstack([Zs[gsel[j]], -np.ones((nP, 1))])
                        out = out + p[3 * k + j] * np.maximum(zeta @ u, 0)
                    return float(np.sum(pi * lossf(out)))
                r = minimize(F, rng.normal(size=4 * k + 1), method="L-BFGS-B")
                best = min(best, r.fun)
            rows.append({"trial": trial, "k": k, "kind": kind, "exact": v, "trained": best})
            viol += int(best < v - 1e-6)
            print(f"trial {trial} k={k} {kind}: exact fold rule {v:.6f}, best of 30 trainings {best:.6f}")
print("training below the exact fold rule:", viol)
ok = viol == 0
print("ALL OK:", ok)
json.dump({"rows": rows, "violations": viol, "all_ok": bool(ok)},
          open(os.path.join(os.path.dirname(__file__), "t27_foldcost.json"), "w"), indent=1)
