# serves: T34
"""Exact computer-assisted proof of Theorem T34 (COARSE-THEORY.md B.Z): signed folds of one family are not optimal.

N: family A, relu(w_i^T x - 1), w_i = (+-1, 0, 1/2), (0, +-1, 1/2); family B, relu(v_k^T x - 1), v_k = (+-1, +-1, -1/2);
gap t = the sum of all eight units. P: uniform on the orbits of (2, 0, 2), (-1, 2, 0), (1, -1, 0) under the rotation by
90 degrees about x_3. D2_P(B) = sum_j pi_j (gap_B - t_j)^2 / 4. Width 1.
Part 1: the build relu(-11/3 x1 + 11/9 x2 - 10/9 x3 - 44/9) + 16/9 has D2_P = 25/216 (exact).
Part 2: the signed folds of one family are the units with |w_3| >= max(|w_1|, |w_2|) / 2 (b free): two convex cones
(tau = +1: w_3 >= 0; tau = -1: w_3 <= 0), each four linear inequalities. Every width-1 build whose unit is one has
D2_P > 25/216, by exact dual certificates over an exact list of pieces (cone, output sign, active set).
Prediction (written before the run): part 1 gives exactly 25/216; every piece of part 2 is certified above 25/216.
"""
import itertools
import json
import os
from fractions import Fraction as Fr

import numpy as np
import sympy as sp
from scipy.optimize import linprog, minimize


def orbit(p):
    out, q = [], tuple(p)
    for _ in range(4):
        out.append(q)
        q = (-q[1], q[0], q[2])
    return out


SEEDS, WTS = [(2, 0, 2), (-1, 2, 0), (1, -1, 0)], [1, 1, 1]
PTS = sum((orbit(s) for s in SEEDS), [])
n = len(PTS)
PI = [sp.Rational(w, 4 * sum(WTS)) for w in WTS for _ in range(4)]  # weights sum to 1
H = sp.Rational(1, 2)
WF = [(1, 0, H), (0, 1, H), (-1, 0, H), (0, -1, H), (1, 1, -H), (-1, 1, -H), (-1, -1, -H), (1, -1, -H)]
relu = lambda v: v if v > 0 else 0
T = [sum(relu(w[0] * a + w[1] * b + w[2] * c - 1) for w in WF) for a, b, c in PTS]
Xt = [[a, b, c, 1] for a, b, c in PTS]
out = {"targets": [str(v) for v in T]}

# ---------- Part 1 ----------
R_ = sp.Rational
gapB = [relu(R_(-11, 3) * a + R_(11, 9) * b - R_(10, 9) * c - R_(44, 9)) + R_(16, 9) for a, b, c in PTS]
d2_1 = sum(PI[j] * (gapB[j] - T[j]) ** 2 for j in range(n)) / 4
TARGET = R_(25, 216)
part1_ok = d2_1 == TARGET
print("targets:", T, "\nPart 1: D2 of the explicit build =", d2_1)
out["part1"] = {"D2": str(d2_1), "ok": bool(part1_ok)}


# ---------- Part 2 ----------
def gordan_exact(rows):
    """Exact: a non-zero non-negative combination of the rows equal to zero (support from a dual LP)."""
    A = np.array(rows, float)
    m = len(rows)
    r = linprog(np.zeros(m), A_eq=np.vstack([A.T, np.ones((1, m))]), b_eq=np.concatenate([np.zeros(A.shape[1]), [1]]),
                bounds=[(0, None)] * m, method="highs")
    if r.status != 0:
        return False
    S = [k for k in range(m) if r.x[k] > 1e-9]
    Ms = sp.Matrix([rows[k] for k in S]).T
    for y in Ms.nullspace():
        if all(v >= 0 for v in y) or all(v <= 0 for v in y):
            return True
    return False


def piece_rows(sg, A):
    """Rows g in R^4 with g . theta <= 0: the closed cell (orthant sg of (w1, w2), active set A) of the broad cone."""
    rows = [[(-1 if i in A else 1) * v for v in Xt[i]] for i in range(n)]
    tau = sg
    rows += [[sp.Rational(1, 2), 0, -tau, 0], [-sp.Rational(1, 2), 0, -tau, 0],
             [0, sp.Rational(1, 2), -tau, 0], [0, -sp.Rational(1, 2), -tau, 0]]
    return rows


def cells():
    res = []
    for sg in (1, -1):
        for signs_p in itertools.product([1, -1], repeat=n):
            strict = [[signs_p[i] * v for v in Xt[i]] for i in range(n)]
            strict += [[-sp.Rational(1, 2), 0, sg, 0], [sp.Rational(1, 2), 0, sg, 0],
                       [0, -sp.Rational(1, 2), sg, 0], [0, sp.Rational(1, 2), sg, 0]]
            Af = np.array([[float(v) for v in row] for row in strict])
            r = linprog(np.zeros(4), A_ub=-Af, b_ub=-np.ones(len(strict)), bounds=[(None, None)] * 4, method="highs")
            if r.status == 0:
                q = [sp.Rational(Fr(float(a)).limit_denominator(10 ** 6).numerator,
                                 Fr(float(a)).limit_denominator(10 ** 6).denominator) for a in r.x]
                if not all(sum(row[c] * q[c] for c in range(4)) > 0 for row in strict):
                    raise SystemExit("rational interior point failed")
                res.append((sg, frozenset(i for i in range(n) if signs_p[i] > 0)))
            else:
                if not gordan_exact(strict):
                    raise SystemExit(f"no exact Gordan certificate for {sg} {signs_p}")
    return res


def in_cone(c, gens):
    if all(ci == 0 for ci in c):
        return True
    cm = sp.Matrix(c)
    for r in (1, 2, 3, 4):
        for S in itertools.combinations(range(len(gens)), r):
            Gs = sp.Matrix([gens[s] for s in S]).T
            if Gs.rank() < r:
                continue
            sol = (Gs.T * Gs).inv() * Gs.T * cm
            if Gs * sol == cm and all(m >= 0 for m in sol):
                return True
    return False


Wd = sp.diag(*PI)
tv = sp.Matrix(T)
Pn = np.array([float(p) for p in PI]); Tn = np.array([float(v) for v in T])


def certify(s, cell):
    sg, A = cell
    L = sp.Matrix([[v if i in A else 0 for v in Xt[i]] for i in range(n)])
    R = piece_rows(sg, A)
    Ln = np.array(L.tolist(), float); Rn = np.array([[float(v) for v in r] for r in R])
    M = np.hstack([s * Ln, np.ones((n, 1))])
    G = np.hstack([Rn, np.zeros((len(R), 1))])
    f = lambda x: float(np.sum(Pn * (M @ x - Tn) ** 2)); df = lambda x: 2 * M.T @ (Pn * (M @ x - Tn))
    best = None
    for start in range(3):
        x0 = np.zeros(5); x0[4] = 0.5
        if start:
            x0[:4] = np.random.default_rng(start).normal(size=4) * 0.1
        r = minimize(f, x0, jac=df, constraints=[{"type": "ineq", "fun": lambda x: -G @ x, "jac": lambda x: -G}],
                     method="SLSQP", options={"maxiter": 3000, "ftol": 1e-15})
        if best is None or r.fun < best.fun:
            best = r
    x = best.x
    sc = max(1.0, np.abs(x).max())
    F = [k for k in range(len(R)) if abs(Rn[k] @ x[:4]) <= 1e-6 * sc]
    Ms = sp.Matrix.hstack(s * L, sp.ones(n, 1))
    E = sp.zeros(len(F), 5)
    for a, k in enumerate(F):
        for c in range(4):
            E[a, c] = R[k][c]
    Nb = E.nullspace() if E.rows else [sp.eye(5)[:, k] for k in range(5)]
    if Nb:
        Nm = sp.Matrix.hstack(*Nb); A_ = Ms * Nm
        y = (A_.T * Wd * A_).pinv() * A_.T * Wd * tv
        v = tv - A_ * y
    else:
        v = tv
    # dual feasibility in the pi-weighted inner product: sum pi v = 0, s L^T (pi v) in cone(R rows)
    pv = Wd * v
    if sum(pv) != 0:
        return False, None
    if not in_cone(list(s * L.T * pv), R):
        return False, None
    if not all(isinstance(e, sp.Rational) for e in v):  # v must be exactly rational
        return False, None
    lb = (2 * (pv.T * tv)[0] - (v.T * Wd * v)[0]) / 4
    return lb > TARGET, lb


cs = cells()
print("signed-fold cells:", len(cs))
lbs, fails = [], []
for s in (1, -1):
    for c in cs:
        ok, lb = certify(s, c)
        (lbs if ok else fails).append(lb if ok else (s, c[0], sorted(c[1])))
print(f"pieces: {2 * len(cs)}; certified above 25/216: {len(lbs)}; not certified: {len(fails)}")
if lbs:
    print("least certified lower bound on D2:", min(lbs), "=", float(min(lbs)))
for f_ in fails[:8]:
    print("  not certified:", f_)
part2_ok = not fails
out["part2"] = {"pieces": 2 * len(cs), "certified": len(lbs), "min_lb": str(min(lbs)) if lbs else None,
                "not_certified": [str(f_) for f_ in fails[:50]], "ok": part2_ok}
out["all_ok"] = bool(part1_ok and part2_ok)
print("ALL OK:", out["all_ok"])
json.dump(out, open(os.path.join(os.path.dirname(__file__), "t34_signed_gap.json"), "w"), indent=1, default=str)
