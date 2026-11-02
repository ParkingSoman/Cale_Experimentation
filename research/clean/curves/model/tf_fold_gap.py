# serves: D25
"""Decisive test of Conjecture D25 (COARSE-THEORY.md D.9): is the least value over covering-fold builds strictly
above the least value over any builds?

tf_clean_optimum.py found least builds (over any units) whose units are not covering folds, but least builds need
not be unique. Here both least values are computed with the exact rules (trunk_exact.exact_min for any units,
trunk_exact.exact_min_fold for covering folds of one family each), on instances small enough for both:
  A: C4, one family, ring of 8 + centre, width 1.
  B: C3, one family, ring of 6 (radius 2), widths 1 and 2.
  C: C3, two families, ring of 6 (radius 2), width 1.
D2 and KL. A gap larger than the solver tolerance (1e-6 relative) means no least build is made of covering folds:
a counterexample to D25 (up to solver accuracy; exact certification would follow for the paper).
Prediction (written before the run): A shows a gap (its least unit fires at the centre, where every member is silent);
B at width 2 shows a gap; the width-1 cases show none.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from trunk_exact import exact_min, exact_min_fold  # noqa: E402


def family(n, offset, bias):
    a = offset + 2 * np.pi * np.arange(n) / n
    return np.stack([np.cos(a), np.sin(a)], 1), np.full(n, bias)


def ring(m, radii, center=False):
    pts = [r * np.array([np.cos(2 * np.pi * j / m), np.sin(2 * np.pi * j / m)]) for r in radii for j in range(m)]
    if center:
        pts.append(np.zeros(2))
    return np.array(pts)


CASES = [
    ("A: C4 one family, 8 ring + centre", [family(4, 0.0, -1.0)], [1.0], ring(8, [2.0], center=True), [1]),
    ("B: C3 one family, 6 ring", [family(3, 0.0, -1.0)], [1.0], ring(6, [2.0]), [1, 2]),
    ("C: C3 two families, 6 ring", [family(3, 0.0, -1.0), family(3, np.pi / 3, -2.0)], [1.0, 0.5], ring(6, [2.0]), [1]),
]
out = {}
for name, fams, ws, X, ks in CASES:
    nP = len(X)
    Zs = [X @ W.T + b for W, b in fams]
    gap = sum(wt * np.maximum(Z, 0).sum(1) for wt, Z in zip(ws, Zs))
    pi = np.full(nP, 1 / nP)
    for k in ks:
        for kind in ("d2", "kl"):
            (v_any, _, _), _ = exact_min(X, pi, gap, k, kind, starts=3)
            v_fold, _, _ = exact_min_fold(Zs, pi, gap, k, kind, starts=3)
            rel = (v_fold - v_any) / max(v_any, 1e-12)
            key = f"{name} | k={k} | {kind}"
            out[key] = {"least_any": v_any, "least_fold": v_fold, "relative_gap": rel}
            print(f"{key}: least over any units {v_any:.6f}; over covering folds {v_fold:.6f}; relative gap {rel:+.2e}")
json.dump(out, open(os.path.join(os.path.dirname(__file__), "tf_fold_gap.json"), "w"), indent=1)
