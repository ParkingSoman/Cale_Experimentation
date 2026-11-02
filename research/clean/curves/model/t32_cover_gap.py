# serves: T32
"""Exact computer-assisted proof of Theorem T32 (COARSE-THEORY.md B.Z): covering folds are not optimal.

N: units relu(x1 - 1), relu(x2 - 1), relu(-x1 - 1), relu(-x2 - 1); gap t = their sum. P: uniform on the eight points
(+-2, 0), (0, +-2), (+-2, +-2). D2_P(B) = (1/32) * sum_j (gap_B(x_j) - t_j)^2. Width 2.
Part 1 (upper bound): an explicit build with rational weights and D2_P <= 3/80, evaluated exactly.
Part 2 (lower bound over covering folds): every width-2 build whose units are covering folds has
sum_j (gap_B - t)^2 >= 4/3, i.e. D2_P >= 1/24 > 3/80.
Method for part 2. Covering folds of this family are relu(theta^T x~) with theta = (w, b), b <= -(|w1| + |w2|)
(Theorem T29). A build is s1 relu(theta1^T x~) + s2 relu(theta2^T x~) + gamma (output sizes absorbed). Pieces: signs
s_j, orthant sigma_j of (w_j1, w_j2), and the strict active set A_j of unit j on P, each realised by a full-dimensional
cell (found by a linear program with margins). On a piece the gap vector is M x with x = (theta1, theta2, gamma) and
the feasible set is the polyhedral cone C = {G x <= 0}. Certificate: if M^T v = G^T lambda with lambda >= 0, then for
every x in C, |M x - t|^2 >= 2 v^T t - |v|^2 (since |a|^2 >= 2 v^T a - |v|^2 and v^T M x = lambda^T G x <= 0). The
script takes v, lambda from the numerical optimum, projects them exactly (rational arithmetic) onto
{M^T v = G^T lambda}, checks lambda >= 0, and checks 2 v^T t - |v|^2 >= 4/3.
The list of cells is exact: each feasible cell is confirmed by a rational interior point, and each infeasible sign
pattern by a Gordan certificate (a positive combination of at most four constraint rows equal to zero).
Prediction (written before the run): part 1 gives exactly 3/80; every piece of part 2 gets a certificate.
"""
import itertools
import json
import os
import sys
from fractions import Fraction as Fr

import numpy as np
import sympy as sp
from scipy.optimize import linprog, minimize

sys.path.insert(0, os.path.dirname(__file__))
from trunk_exact import exact_min  # noqa: E402

PTS = [(2, 0), (0, 2), (-2, 0), (0, -2), (2, 2), (-2, 2), (-2, -2), (2, -2)]
n = len(PTS)
Xt = np.array([[a, b, 1] for a, b in PTS], float)
WF = [(1, 0), (0, 1), (-1, 0), (0, -1)]
t = np.array([sum(max(w[0] * a + w[1] * b - 1, 0) for w in WF) for a, b in PTS], float)
tF = [sp.Integer(int(v)) for v in t]
BOUND = sp.Rational(4, 3)
out = {}

# ---------- Part 1: explicit build ----------
# the least build found by the exact rule (trunk_exact.exact_min), written with rational weights:
# gap_B(x) = relu(-4/5 x1 + 4/5 x2 - 8/5) - relu(-3/10 x1 + 3/10 x2) + 8/5
relu = lambda v: v if v > 0 else 0
TH = [(sp.Rational(-4, 5), sp.Rational(4, 5), sp.Rational(-8, 5)), (sp.Rational(-3, 10), sp.Rational(3, 10), 0)]
SG = [1, -1]
GAM = sp.Rational(8, 5)
gapB = [sum(SG[j] * relu(TH[j][0] * a + TH[j][1] * b + TH[j][2]) for j in range(2)) + GAM for a, b in PTS]
d2_exact = sp.nsimplify(sum((gapB[i] - int(t[i])) ** 2 for i in range(n)) / 32)
(v_any, _, _), _ = exact_min(Xt[:, :2], np.full(n, 1 / n), t, 2, "d2", starts=4)
part1_ok = d2_exact == sp.Rational(3, 80)
print(f"Part 1: numerical least over any builds {v_any:.6f}; explicit rational build D2 = {d2_exact}")
out["part1"] = {"build": "relu(-4/5 x1 + 4/5 x2 - 8/5) - relu(-3/10 x1 + 3/10 x2) + 8/5", "D2": str(d2_exact),
                "ok": bool(part1_ok)}


# ---------- Part 2: pieces and certificates ----------
def gordan(rows):
    """Exact: a non-zero non-negative combination of at most four integer rows (in R^3) that equals zero."""
    R = [tuple(int(v) for v in r) for r in rows]
    for r in (1, 2, 3, 4):
        for S in itertools.combinations(range(len(R)), r):
            A = sp.Matrix([R[s] for s in S]).T  # 3 x r
            for y in A.nullspace():
                if all(v >= 0 for v in y) or all(v <= 0 for v in y):
                    return True
    return False


def cells():
    """(sigma, A) with a full-dimensional covering cell, listed exactly."""
    res = []
    for sg in itertools.product([1, -1], repeat=2):
        for signs_p in itertools.product([1, -1], repeat=n):
            sv = np.array(signs_p, float)
            strict = [list((sv[i] * Xt[i]).astype(int)) for i in range(n)]
            strict += [[sg[0], 0, 0], [0, sg[1], 0], [-sg[0], -sg[1], -1]]  # interior of K_sigma
            r = linprog(np.zeros(3), A_ub=-np.array(strict, float), b_ub=-np.ones(len(strict)),
                        bounds=[(None, None)] * 3, method="highs")
            if r.status == 0:
                q = [sp.Rational(Fr(float(a)).limit_denominator(10 ** 6).numerator,
                                 Fr(float(a)).limit_denominator(10 ** 6).denominator) for a in r.x]
                if not all(sum(row[c] * q[c] for c in range(3)) > 0 for row in strict):
                    raise SystemExit(f"cell {sg} {signs_p}: rational interior point failed")
                res.append((sg, frozenset(np.where(sv > 0)[0])))
            else:
                if not gordan(strict):
                    raise SystemExit(f"cell {sg} {signs_p}: no exact Gordan certificate")
    return res


def in_cone(c, gens):
    """Exact: is the 3-vector c a non-negative combination of the rows gens (Caratheodory: at most 3 at a time)?"""
    if all(ci == 0 for ci in c):
        return True
    cm = sp.Matrix(c)
    for r in (1, 2, 3):
        for S in itertools.combinations(range(len(gens)), r):
            Gs = sp.Matrix([gens[s] for s in S]).T  # 3 x r
            if Gs.rank() < r:
                continue
            sol = (Gs.T * Gs).inv() * Gs.T * cm
            if Gs * sol == cm and all(m >= 0 for m in sol):
                return True
    return False


def unit_rows(sg, A):
    """Rows g in R^3 with g . theta <= 0 describing the closed covering cell (sigma, A)."""
    rows = [[(-1 if i in A else 1) * int(Xt[i, c]) for c in range(3)] for i in range(n)]
    rows += [[-sg[0], 0, 0], [0, -sg[1], 0], [sg[0], sg[1], 1]]
    return rows


def certify(s1, c1, s2, c2):
    (sg1, A1), (sg2, A2) = c1, c2
    L = []
    for A in (A1, A2):
        L.append(sp.Matrix([[int(Xt[i, c]) if i in A else 0 for c in range(3)] for i in range(n)]))
    R1, R2 = unit_rows(sg1, A1), unit_rows(sg2, A2)
    # numerical optimum of the piece
    M = np.zeros((n, 7)); M[:, 0:3] = s1 * np.array(L[0].tolist(), float); M[:, 3:6] = s2 * np.array(L[1].tolist(), float)
    M[:, 6] = 1.0
    G = np.zeros((len(R1) + len(R2), 7)); G[:len(R1), 0:3] = R1; G[len(R1):, 3:6] = R2
    f = lambda x: float(np.sum((M @ x - t) ** 2)); df = lambda x: 2 * M.T @ (M @ x - t)
    con = {"type": "ineq", "fun": lambda x: -G @ x, "jac": lambda x: -G}
    best = None
    for start in range(3):
        x0 = np.zeros(7); x0[6] = 1.5
        if start:
            x0[:6] = np.random.default_rng(start).normal(size=6) * 0.1
        r = minimize(f, x0, jac=df, constraints=[con], method="SLSQP", options={"maxiter": 3000, "ftol": 1e-15})
        if best is None or r.fun < best.fun:
            best = r
    x = best.x
    scale = max(1.0, np.abs(x).max())
    F1 = [k for k in range(len(R1)) if abs(np.dot(R1[k], x[0:3])) <= 1e-6 * scale]
    F2 = [k for k in range(len(R2)) if abs(np.dot(R2[k], x[3:6])) <= 1e-6 * scale]
    # exact least squares on the face {R1[F1] theta1 = 0, R2[F2] theta2 = 0}
    blocks = []
    E = sp.zeros(len(F1) + len(F2), 7)
    for a, k in enumerate(F1):
        for c in range(3):
            E[a, c] = R1[k][c]
    for a, k in enumerate(F2):
        for c in range(3):
            E[len(F1) + a, 3 + c] = R2[k][c]
    Ms = sp.Matrix.hstack(s1 * L[0], s2 * L[1], sp.ones(n, 1))
    Nb = E.nullspace() if E.rows else [sp.eye(7)[:, k] for k in range(7)]
    if not Nb:
        v = sp.Matrix([int(a) for a in t])
    else:
        Nm = sp.Matrix.hstack(*Nb)
        A_ = Ms * Nm
        y = (A_.T * A_).pinv() * A_.T * sp.Matrix([int(a) for a in t])
        v = sp.Matrix([int(a) for a in t]) - A_ * y
    # dual feasibility: 1 . v = 0 and s_j L_j^T v in cone(rows of unit j)
    if sum(v) != 0:
        return False, None
    c1v = list(s1 * L[0].T * v); c2v = list(s2 * L[1].T * v)
    if not (in_cone(c1v, R1) and in_cone(c2v, R2)):
        return False, None
    if not all(isinstance(e, sp.Rational) for e in v):  # v must be exactly rational
        return False, None
    lb = 2 * sum(v[i] * tF[i] for i in range(n)) - sum(v[i] ** 2 for i in range(n))
    return lb >= BOUND, lb


cs = cells()
print("covering cells (sigma, active set):", len(cs))
units = [(s, c) for c in cs for s in (1, -1)]
fails = []
lbs = []
for i, (s1, c1) in enumerate(units):
    for (s2, c2) in units[i:]:
        ok, lb = certify(s1, c1, s2, c2)
        if not ok:
            fails.append((s1, sorted(c1[1]), c1[0], s2, sorted(c2[1]), c2[0], str(lb)))
        else:
            lbs.append(lb)
print(f"pieces: {len(units) * (len(units) + 1) // 2}; certified >= 4/3: {len(lbs)}; not certified: {len(fails)}")
if lbs:
    print("smallest certified lower bound (sum of squares):", min(lbs), "=", float(min(lbs)))
for f_ in fails[:10]:
    print("  not certified:", f_)
part2_ok = not fails
out["part2"] = {"pieces": len(units) * (len(units) + 1) // 2, "certified": len(lbs), "not_certified": fails[:50],
                "min_lb": str(min(lbs)) if lbs else None, "ok": part2_ok}
ALL = part1_ok and part2_ok
out["all_ok"] = ALL
print("ALL OK:", ALL)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "t32_cover_gap.json"), "w"), indent=1, default=str)
