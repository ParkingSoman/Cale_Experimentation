# serves: TH3
"""Proposition TH3 (COARSE-THEORY.md B.H): at a lossy sibling merge with three logits, step by step is strictly worse.
Exact part (fractions, sympy): claims at error 0; Sigma_1, Sigma_2 and their eigenvalues; the builds of the proof give
D2 = 3/2 at G_1, 2 at G_2, and 3 for the step that copies child 1's unit. Numeric part: least D2 searched over
(a) one box unit of S_1 alone (expect 3/2), (b) one box unit of the merged group (expect 2), (c) every (r) step that
reads the least B_1's units relu(x_a), relu(x_c) (expect 3, never below)."""
import json
import os
from fractions import Fraction as Fr

import numpy as np
import sympy as sp
from scipy.optimize import minimize


def relu(t):
    return t if t > 0 else 0 * t


ea, eb = [Fr(1), Fr(-1), Fr(0)], [Fr(1), Fr(1), Fr(-2)]
pts, wts, area = [], [], []
for sa, sb in ((2, 1), (2, -1), (-2, 1), (-2, -1)):
    pts.append([Fr(1), Fr(0), Fr(3 + sa), Fr(3 + sb), Fr(-1)]); wts.append(Fr(1, 8)); area.append(1)
for sc in (1, -1):
    pts.append([Fr(0), Fr(1), Fr(-1), Fr(-1), Fr(3 + sc)]); wts.append(Fr(1, 4)); area.append(2)
n = len(pts)
H = [[relu(v) for v in x] for x in pts]          # units J1, J2, u_a, u_b, u_c (identity weights)
yN = [[ea[i] * h[2] + eb[i] * h[3] + eb[i] * h[4] for i in range(3)] for h in H]


def Pperp(v):
    m = sum(v) / 3
    return [a - m for a in v]


def D2(yB):
    return Fr(1, 2) * sum(w * sum(d * d for d in Pperp([a - b for a, b in zip(yb, yn)])) for w, yb, yn in zip(wts, yB, yN))


# claims
claims = {}
for name, u, ar in (("u_a", 2, 1), ("u_b", 3, 1), ("u_c", 4, 2), ("J1", 0, 1), ("J2", 1, 2)):
    out = [k for k in range(n) if area[k] != ar]
    claims[name + " district"] = sum(wts[k] * H[k][u] for k in out)
    claims[name + " shelter_nonconst"] = Fr(len({pts[k][u] for k in out}) > 1)
for J, ar in ((0, 1), (1, 2)):
    claims[f"judgment J{J + 1}"] = sum(wts[k] * (1 - H[k][J]) ** 2 for k in range(n) if area[k] == ar)
claims_ok = all(v == 0 for v in claims.values())

# Sigma matrices (centred signal per area)
def sigma(ar):
    ks = [k for k in range(n) if area[k] == ar]
    mean = [sum(wts[k] * yN[k][i] for k in ks) / sum(wts[k] for k in ks) for i in range(3)]
    return sp.Matrix(3, 3, lambda i, j: sum(wts[k] * (yN[k][i] - mean[i]) * (yN[k][j] - mean[j]) for k in ks))


S1, S2 = sigma(1), sigma(2)
ev1, ev12 = S1.eigenvals(), (S1 + S2).eigenvals()
print("eigenvalues Sigma_1:", ev1, " Sigma_1 + Sigma_2:", ev12, " traces:", S1.trace(), S2.trace())


def build(units, V, J1w, J2w, bias):
    """units: list of per-point unit values; V: list of 3-vectors."""
    return [[sum(V[t][i] * units[t][k] for t in range(len(units))) + J1w[i] * H[k][0] + J2w[i] * H[k][1] + bias[i]
             for i in range(3)] for k in range(n)]


def fit_offsets(units, V):
    """Exact least offsets: on each area add the mean residual (J weights), bias 0 (no point is outside both)."""
    J = {}
    for ar, col in ((1, 0), (2, 1)):
        ks = [k for k in range(n) if area[k] == ar]
        W = sum(wts[k] for k in ks)
        J[ar] = [sum(wts[k] * (yN[k][i] - sum(V[t][i] * units[t][k] for t in range(len(units)))) for k in ks) / W
                 for i in range(3)]
    return build(units, V, J[1], J[2], [Fr(0)] * 3)


ua = [relu(x[2]) for x in pts]
uc = [relu(x[4]) for x in pts]
ubc = [relu(x[3] + x[4]) for x in pts]
d_G1 = D2(fit_offsets([ua, uc], [ea, eb]))
d_G2 = D2(fit_offsets([ubc], [eb]))
d_copy = D2(fit_offsets([ua], [ea]))
print("D2 at G_1 (relu(x_a)->e_a, relu(x_c)->e_b):", d_G1, " at G_2 (relu(x_b + x_c)->e_b):", d_G2,
      " copy step (relu(x_a)->e_a):", d_copy)
exact_ok = claims_ok and d_G1 == Fr(3, 2) and d_G2 == 2 and d_copy == 3 and \
    sorted(ev1.keys()) == [0, 3, 4] and sorted(ev12.keys()) == [0, 4, 6]

# numeric searches
X = np.array([[float(v) for v in x] for x in pts]); Wt = np.array([float(w) for w in wts]); A = np.array(area)
Y = np.array([[float(v) for v in y] for y in yN])


def d2_of(unit_vals, mask=None):
    """Least D2 given one unit's values: free output vector v and free per-area offsets (closed form in v)."""
    def obj(v):
        tot = 0.0
        for ar in (1, 2):
            m = (A == ar) if mask is None else ((A == ar) & mask)
            if not m.any():
                continue
            R = Y[m] - np.outer(unit_vals[m], v)
            R = R - (Wt[m, None] * R).sum(0) / Wt[m].sum()
            R = R - R.mean(1, keepdims=True)
            tot += 0.5 * (Wt[m, None] * R ** 2).sum()
        return tot
    return min(minimize(obj, np.random.default_rng(s).normal(size=3) * 2).fun for s in range(4))


rng = np.random.default_rng(1)
relu_np = lambda t: np.maximum(t, 0)
best_child = min(d2_of(relu_np(X[:, 2:4] @ m + b), mask=(A == 1))
                 for m, b in ((rng.normal(size=2) * 2, rng.normal() * 3) for _ in range(300)))
def opt_unit(cols, mask=None, starts=40):
    f = lambda th: d2_of(relu_np(X[:, cols] @ th[:-1] + th[-1]), mask=mask)
    return min(minimize(f, np.r_[rng.normal(size=len(cols)) * 2, abs(rng.normal()) * 4], method="Nelder-Mead",
                        options=dict(maxiter=600)).fun for _ in range(starts))


best_merge = opt_unit([2, 3, 4])
p1 = X[:, 2]; p2 = X[:, 4]          # pre-activations of B_1's least units (any positive rescaling is the same)
best_r = min(d2_of(relu_np(mu1 * (s1 * p1 + o1) + mu2 * p2 + b))
             for mu1, mu2, b, s1, o1 in ((rng.normal() * 2, rng.normal() * 2, rng.normal() * 4, rng.choice([-1, 1]),
                                          rng.normal() * 3) for _ in range(600)))
print(f"numeric: child 1 alone {best_child:.6f} (exact 3/2), merged group {best_merge:.6f} (exact 2), "
      f"every (r) step >= {best_r:.6f} (proved 3)")
num_ok = abs(best_child - 1.5) < 1e-3 and abs(best_merge - 2) < 1e-3 and best_r > 3 - 1e-6
ALL_OK = bool(exact_ok and num_ok)
print("CLAIMS_OK", claims_ok, "EXACT_OK", exact_ok, "ALL_OK", ALL_OK)
with open(os.path.join(os.path.dirname(__file__), "th_break.json"), "w") as f:
    json.dump(dict(claims={k: str(v) for k, v in claims.items()}, eig_S1=str(ev1), eig_S12=str(ev12),
                   D2_G1=str(d_G1), D2_G2=str(d_G2), D2_copy=str(d_copy), num_child=best_child, num_merge=best_merge,
                   num_r=best_r, ALL_OK=ALL_OK), f, indent=1)
