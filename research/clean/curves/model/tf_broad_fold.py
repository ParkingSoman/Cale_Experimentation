# serves: D26
"""Exact test of Conjecture D26 (COARSE-THEORY.md D.9): the least build is made of broad folds.

d = 3. One family, an orbit of C3 (rotation about the x_3 axis): units relu(w_i^T x - 1) with
w_i = (cos a_i, sin a_i, h), a_i = 0, 120, 240 degrees, h > 0, so every member leans up and the cone of the w_i is a
proper cone (every non-negative mix has a positive x_3 component). The w_i span R^3 (full rank, Lemma T31).
Mix (second version, twelve points; the six-point version was degenerate: one unit copied N exactly): at height +c,
radius r at angles 0, 120, 240 and radius r/2 at 60, 180, 300; at height -c, radius r at 60, 180, 300 and r/2 at
0, 120, 240. Invariant under the C3 rotation. Gap = sum of the units. Width 1 (width 2 is too slow for the exact
fold rule at twelve points); D2 and KL.
Three least values by the exact rules of trunk_exact.py: over any builds (exact_min), over broad folds
(exact_min_fold with a free offset: the group's pre-activations plus a constant column), and over covering folds
(exact_min_fold as is).
Prediction (written before the run): any = broad (no gap) in every case; covering folds show a gap.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from trunk_exact import exact_min, exact_min_fold  # noqa: E402

out = {}
for (h, r, c) in [(0.5, 2.0, 1.0), (0.3, 1.5, 2.0)]:
    a = 2 * np.pi * np.arange(3) / 3
    W = np.stack([np.cos(a), np.sin(a), np.full(3, h)], 1)
    b = np.full(3, -1.0)
    a6 = 2 * np.pi * np.arange(6) / 6
    up = [rr * np.array([np.cos(t), np.sin(t), 0]) + np.array([0, 0, c]) for rr in (r, r / 2) for t in a6[::2] + (0 if rr == r else np.pi / 3)]
    down = [rr * np.array([np.cos(t), np.sin(t), 0]) - np.array([0, 0, c]) for rr in (r, r / 2) for t in a6[1::2] + (0 if rr == r else np.pi / 3)]
    X = np.array(up + down)
    Z = X @ W.T + b
    gap = np.maximum(Z, 0).sum(1)
    pi = np.full(len(X), 1 / len(X))
    Zbroad = np.hstack([Z, np.ones((len(X), 1))])  # with zeta's -1 column the offset becomes free
    for k in (1,):
        for kind in ("d2", "kl"):
            (v_any, _, _), _ = exact_min(X, pi, gap, k, kind, starts=3)
            v_broad, _, _ = exact_min_fold([Zbroad], pi, gap, k, kind, starts=3)
            v_cov, _, _ = exact_min_fold([Z], pi, gap, k, kind, starts=3)
            key = f"h={h} r={r} c={c} | k={k} | {kind}"
            out[key] = {"any": v_any, "broad": v_broad, "covering": v_cov}
            print(f"{key}: any {v_any:.6f}  broad folds {v_broad:.6f} (gap {(v_broad - v_any) / max(v_any, 1e-12):+.1e})"
                  f"  covering folds {v_cov:.6f} (gap {(v_cov - v_any) / max(v_any, 1e-12):+.1e})")
json.dump(out, open(os.path.join(os.path.dirname(__file__), "tf_broad_fold.json"), "w"), indent=1)
