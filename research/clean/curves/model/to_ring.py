# serves: D29
# serves: D30
"""Exact test of Conjectures D29 and D30 (COARSE-THEORY.md D.10): the least width-1 unit of one family on rings.

Family: relu(u_i^T x - 1), u_i at angles 2 pi i / n. Mix: rings of points at angles multiple of pi / n (invariant
under the dihedral group D_n), optional centre, weights per ring. Two logits, gap = sum of the units. Width 1.
For D2 and KL:
  least_all  : exact rule over all width-1 builds (trunk_exact.exact_min, d = 2)
  least_axis : the least over the n mirror axes of the exact rule along the axis (one-dimensional inputs u . x)
  least_quiet: the exact rule over width-1 builds whose unit is silent at every mix point where all members are silent
D29 holds on an instance when least_axis = least_all; D30 when least_quiet > least_all (if silent points exist).
Values at solver accuracy (exact enumeration of activation patterns; convex pieces by SLSQP).
Prediction (written before the run): D29 holds on every instance; D30 holds whenever the mix has silent points.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from trunk_exact import exact_min, per_point_loss, regions, solve_piece  # noqa: E402


def instance(n, radii, center, ring_w):
    U = np.stack([np.cos(2 * np.pi * np.arange(n) / n), np.sin(2 * np.pi * np.arange(n) / n)], 1)
    pts, wts = [], []
    for r, wr in zip(radii, ring_w):
        for j in range(2 * n):
            a = np.pi * j / n
            pts.append([r * np.cos(a), r * np.sin(a)]); wts.append(wr)
    if center:
        pts.append([0.0, 0.0]); wts.append(ring_w[0])
    X = np.array(pts); w = np.array(wts, float)
    Z = X @ U.T - 1.0
    return X, w / w.sum(), np.maximum(Z, 0).sum(1), Z


def least_quiet(X, pi, t, kind, quiet):
    regs, Xt = regions(X)
    lossf, gradf = per_point_loss(kind, np.asarray(t, float))
    best = np.inf
    for A in regs:
        if A & quiet:
            continue
        for s in (1, -1):
            v, _ = solve_piece(Xt, pi, lossf, gradf, [(s, A)], starts=3)
            best = min(best, v)
    return best


CASES = [
    (3, [1.5, 3.0], False, [1, 1]), (3, [1.5, 3.0], True, [1, 1]), (3, [2.0], True, [1]), (3, [1.5, 3.0], True, [3, 1]),
    (4, [2.0], True, [1]), (4, [1.5], True, [1]), (4, [3.0], True, [2]),
    (5, [2.0], True, [1]), (5, [1.5], False, [1]),
    (6, [2.0], False, [1]), (6, [1.5], False, [1]),
]
out = {}
d29_ok = True
d30_ok = True
for (n, radii, center, ring_w) in CASES:
    X, pi, t, Z = instance(n, radii, center, ring_w)
    quiet = frozenset(np.where((Z <= 1e-12).all(1))[0])
    for kind in ("d2", "kl"):
        (v_all, _, _), _ = exact_min(X, pi, t, 1, kind, starts=3)
        v_axis = np.inf
        for j in range(n):
            u = np.array([np.cos(np.pi * j / n), np.sin(np.pi * j / n)])
            (v, _, _), _ = exact_min((X @ u)[:, None], pi, t, 1, kind, starts=3)
            v_axis = min(v_axis, v)
        v_quiet = least_quiet(X, pi, t, kind, quiet) if quiet else None
        tol = 1e-6 * max(1.0, v_all)
        axis_holds = v_axis <= v_all + tol
        quiet_holds = (v_quiet > v_all + tol) if quiet else None
        d29_ok = d29_ok and axis_holds
        if quiet:
            d30_ok = d30_ok and quiet_holds
        key = f"n={n} radii={radii} centre={center} w={ring_w} {kind}"
        out[key] = {"least_all": v_all, "least_axis": v_axis, "least_quiet": v_quiet, "n_quiet": len(quiet),
                    "D29": bool(axis_holds), "D30": quiet_holds}
        print(f"{key}: all {v_all:.6f} axis {v_axis:.6f} (D29 {axis_holds}) "
              + (f"quiet {v_quiet:.6f} on {len(quiet)} silent points (D30 {quiet_holds})" if quiet else "no silent points"),
              flush=True)
out["D29_all"] = bool(d29_ok); out["D30_all"] = bool(d30_ok)
print("D29 holds on every instance:", d29_ok, "| D30 holds on every instance with silent points:", d30_ok)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "to_ring.json"), "w"), indent=1, default=str)
