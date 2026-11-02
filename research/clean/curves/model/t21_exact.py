# serves: T21
"""Check of Theorems T19 and T21 (COARSE-THEORY.md B.T) on small random instances.

For random finite mixes (n = 6 points, d = 1 and d = 2) and random teacher gaps, compare the exact rule of T21
(trunk_exact.exact_min: regions x signs x convex pieces) with ordinary gradient training of the same build
(L-BFGS on the raw weights, 30 random starts), for k = 1 and k = 2, objectives KL and D2.
Prediction (written before the run): training never ends below the exact rule (beyond solver tolerance 1e-6), and on
some instances every training start ends above it.
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
from trunk_exact import exact_min, per_point_loss  # noqa: E402


def train(X, pi, t, k, kind, starts=30, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    lossf, _ = per_point_loss(kind, t)

    def F(z):
        W = z[:k * d].reshape(k, d)
        b = z[k * d:k * d + k]
        a = z[k * d + k:k * d + 2 * k]
        g = z[-1]
        f = np.maximum(X @ W.T + b, 0) @ a + g
        return float(np.sum(pi * lossf(f)))

    best = np.inf
    for _ in range(starts):
        z0 = rng.normal(size=k * d + 2 * k + 1)
        r = minimize(F, z0, method="L-BFGS-B", options={"maxiter": 5000})
        best = min(best, r.fun)
    return best


rows = []
viol = 0
stuck = 0
rng = np.random.default_rng(1)
for trial in range(8):
    d = 1 if trial < 4 else 2
    n = 6
    X = rng.normal(size=(n, d)).round(2)
    pi = np.full(n, 1 / n)
    t = rng.normal(scale=2.0, size=n).round(2)
    for k in (1, 2):
        for kind in ("kl", "d2"):
            (v, z, ch), _ = exact_min(X, pi, t, k, kind, starts=3)
            tr = train(X, pi, t, k, kind)
            rows.append({"trial": trial, "d": d, "k": k, "kind": kind, "exact": v, "trained": tr})
            if tr < v - 1e-6:
                viol += 1
            if tr > v + 1e-4:
                stuck += 1
            print(f"trial {trial} d={d} k={k} {kind}: exact {v:.6f}  best of 30 trainings {tr:.6f}")
print(f"training below exact rule: {viol} cases; training stuck above (by > 1e-4): {stuck} of {len(rows)}")
json.dump({"rows": rows, "violations": viol, "stuck": stuck},
          open(os.path.join(os.path.dirname(__file__), "t21_exact.json"), "w"), indent=1)
