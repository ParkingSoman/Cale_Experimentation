# serves: D25
"""Exact test of Conjecture D25 (COARSE-THEORY.md D.9): for clean networks, the least build is a fold.

Clean networks in R^2 (d = 2, so the first layer spans R^2 and Lemma T31 applies), two logits, gap = weighted sum of
each family's units. Families are orbits of the rotation group C_n: units relu(cos(a_i) x_1 + sin(a_i) x_2 + b).
Mixes: rings of points whose angles are invariant under the group, equal weights.
For each instance, width k and objective (D2, KL): the exact any-unit rule of Theorem T21 (trunk_exact.exact_min),
then for every unit of the least build a linear program: is its data vector on P equal (up to the scale already in
the solution) to that of a covering fold of ONE family, relu(sum m_i z_i - tau), m >= 0, tau >= 0?
Prediction (written before the run): Conjecture D25 holds on every instance (every least unit is a covering fold of
one family); if it fails, the failing units are reported with their best fit.
Note: the exact rule returns one least build; a failure means that build's units are not covering folds, and other
least builds (not images under the group) would still need checking before calling it a counterexample.
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(__file__))
from trunk_exact import exact_min  # noqa: E402


def family(n, offset, bias):
    a = offset + 2 * np.pi * np.arange(n) / n
    return np.stack([np.cos(a), np.sin(a)], 1), np.full(n, bias)


def ring(m, radii, offset=0.0, center=False):
    pts = [r * np.array([np.cos(offset + 2 * np.pi * j / m), np.sin(offset + 2 * np.pi * j / m)])
           for r in radii for j in range(m)]
    if center:
        pts.append(np.zeros(2))
    return np.array(pts)


INSTANCES = {
    "C3 one family, 6x2 ring": ([family(3, 0.0, -1.0)], [1.0], ring(6, [1.5, 3.0])),
    "C4 one family, 8 ring + centre": ([family(4, 0.0, -1.0)], [1.0], ring(8, [2.0], center=True)),
    "C6 one family, 12 ring": ([family(6, 0.0, -1.0)], [1.0], ring(12, [2.0])),
    "C3 two families, 6x2 ring": ([family(3, 0.0, -1.0), family(3, np.pi / 3, -2.0)], [1.0, 0.5],
                                  ring(6, [1.5, 3.0])),
}


def covering_fold_residual(v, Z):
    """min sum |sum_i m_i Z[j,i] - tau - v_j| over active j, subject to sum_i m_i Z[j,i] - tau <= 0 on inactive j,
    m, tau >= 0. Z: (n_P x s) family pre-activations. Returns the least total deviation (0 means a covering fold)."""
    nP, s = Z.shape
    act = v > 1e-7 * max(1.0, np.abs(v).max())
    na = int(act.sum())
    # variables: m (s), tau (1), e+ (na), e- (na)
    nv = s + 1 + 2 * na
    c = np.concatenate([np.zeros(s + 1), np.ones(2 * na)])
    A_eq = np.zeros((na, nv)); b_eq = np.zeros(na)
    for r, j in enumerate(np.where(act)[0]):
        A_eq[r, :s] = Z[j]; A_eq[r, s] = -1.0
        A_eq[r, s + 1 + r] = -1.0; A_eq[r, s + 1 + na + r] = 1.0
        b_eq[r] = v[j]
    inact = np.where(~act)[0]
    A_ub = np.zeros((len(inact), nv)); b_ub = np.zeros(len(inact))
    for r, j in enumerate(inact):
        A_ub[r, :s] = Z[j]; A_ub[r, s] = -1.0
    res = linprog(c, A_ub=A_ub if len(inact) else None, b_ub=b_ub if len(inact) else None,
                  A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * nv, method="highs")
    return res.fun if res.status == 0 else np.inf


out = {}
all_fold = True
for name, (fams, ws, X) in INSTANCES.items():
    nP = len(X)
    Zf = [X @ W.T + b for W, b in fams]
    gap = sum(wt * np.maximum(Z, 0).sum(1) for wt, Z in zip(ws, Zf))
    pi = np.full(nP, 1 / nP)
    for k in (1, 2):
        for kind in ("d2", "kl"):
            if k == 2 and nP > 9:
                starts = 1
            else:
                starts = 2
            (v, z, ch), Xt = exact_min(X, pi, gap, k, kind, starts=starts)
            D = Xt.shape[1]
            units = []
            for j, (s, A) in enumerate(ch):
                th = z[j * D:(j + 1) * D]
                vec = np.where([i in A for i in range(nP)], Xt @ th, 0.0)
                if np.abs(vec).max() < 1e-9:
                    units.append({"unit": j, "zero": True})
                    continue
                res = [covering_fold_residual(vec, Z) for Z in Zf]
                best_f = int(np.argmin(res))
                rel = res[best_f] / max(1e-12, np.abs(vec).sum())
                ok = rel < 1e-4
                all_fold = all_fold and ok
                units.append({"unit": j, "sign": s, "theta": th.round(5).tolist(), "best_family": best_f,
                              "relative_residual": rel, "covering_fold": bool(ok)})
            key = f"{name} | k={k} | {kind}"
            out[key] = {"least_value": v, "units": units}
            verdict = all(u.get("covering_fold", True) for u in units)
            print(f"{key}: least {v:.6f}; units covering folds of one family: {verdict}  "
                  + "; ".join(f"u{u['unit']}: fam {u.get('best_family')} res {u.get('relative_residual', 0):.1e}"
                              for u in units if not u.get("zero")))
out["all_covering_folds"] = bool(all_fold)
print("Conjecture D25 holds on every instance (for the returned least builds):", all_fold)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "tf_clean_optimum.json"), "w"), indent=1, default=float)
