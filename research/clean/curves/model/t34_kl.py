# serves: T34
"""Exact (interval-certified) proof of Theorem T34 for KL (COARSE-THEORY.md B.F), extending t34_signed_gap.py.

Same network, mix and pieces as t34_signed_gap.py. Two logits; the KL score is D_P(B) = sum_i pi_i l(g_i; t_i) with
l(f; t) = log(1 + e^f) - q f - H(q), q = sigma(t). Fenchel: log(1 + e^f) >= s f + H(s) for s in (0, 1). So for any v
with s = q - v in (0, 1)^n: D_P >= sum pi (s - q) g + sum pi (H(s) - H(q)) = -<pi v, g> + sum pi (H(q - v) - H(q)).
If sum pi v = 0 and s_j L_j^T (pi v) is in the cone of the rows of R_j (Lemma T32a part 1's condition), then
<pi v, g> <= 0 on the piece, and D_P >= sum pi (H(q - v) - H(q)). Here v is rational (exact conditions, checked as in
Lemma T32a), and the bound is evaluated with interval arithmetic (mpmath, 50 digits).
Part 1 (upper bound): the KL value of an explicit rational build, enclosed by interval arithmetic.
Part 2 (lower bound): every piece of signed folds of one family gets a certificate above part 1's upper bound.
Prediction (written before the run): every piece is certified.
"""
import itertools
import json
import os
import sys
from fractions import Fraction as Fr

import numpy as np
import sympy as sp
from mpmath import iv
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
import t34_signed_gap as base  # noqa: E402  (reuses the exact cell list, rows, points, targets)

iv.dps = 50
n = base.n
PI = base.PI
Tn = [int(v) for v in base.T]
Xt = base.Xt
rat = lambda a: sp.Rational(Fr(float(a)).limit_denominator(10 ** 9).numerator, Fr(float(a)).limit_denominator(10 ** 9).denominator)
ivr = lambda q: iv.mpf(int(sp.numer(q))) / int(sp.denom(q))
qiv = [1 / (1 + iv.exp(-iv.mpf(tt))) for tt in Tn]
H = lambda s: -s * iv.log(s) - (1 - s) * iv.log(1 - s)


def kl_of(gvals):
    tot = iv.mpf(0)
    for i in range(n):
        f = gvals[i]
        tot += ivr(PI[i]) * (iv.log(1 + iv.exp(f)) - qiv[i] * f - H(qiv[i]))
    return tot


# ---------- Part 1: the explicit build (numerical KL optimum over any units, rounded) ----------
sys.path.insert(0, os.path.dirname(__file__))
from trunk_exact import exact_min  # noqa: E402
X3 = np.array([[float(v) for v in r[:3]] for r in Xt])
pin = np.array([float(p) for p in PI])
(vk, zk, chk), _ = exact_min(X3, pin, np.array(Tn, float), 1, "kl", starts=4)
th = [rat(a) for a in zk[:4]]; gam = rat(zk[-1]); sgn = chk[0][0]
gB = [sgn * max(sum(th[c] * Xt[i][c] for c in range(4)), 0) + gam for i in range(n)]
UB = kl_of([ivr(g) for g in gB]).b
print(f"Part 1: numerical least KL over any builds {vk:.8f}; explicit rational build KL <= {float(UB):.8f}")


# ---------- Part 2 ----------
def certify_kl(s, cell):
    sg, A = cell
    L = sp.Matrix([[v if i in A else 0 for v in Xt[i]] for i in range(n)])
    R = base.piece_rows(sg, A)
    Ln = np.array(L.tolist(), float); Rn = np.array([[float(v) for v in r] for r in R])
    M = np.hstack([s * Ln, np.ones((n, 1))]); G = np.hstack([Rn, np.zeros((len(R), 1))])
    q = 1 / (1 + np.exp(-np.array(Tn, float)))
    f = lambda x: float(np.sum(pin * (np.logaddexp(0, M @ x) - q * (M @ x))))
    df = lambda x: M.T @ (pin * (1 / (1 + np.exp(-(M @ x))) - q))
    sols = []
    for start in range(8):
        x0 = np.zeros(5); x0[4] = [1.5, 2.0, 1.0, 2.5][start % 4]
        if start:
            x0[:4] = np.random.default_rng(start).normal(size=4) * (0.1 if start < 4 else 1.0)
        r = minimize(f, x0, jac=df, constraints=[{"type": "ineq", "fun": lambda x: -G @ x, "jac": lambda x: -G}],
                     method="SLSQP", options={"maxiter": 5000, "ftol": 1e-15})
        if np.all(G @ r.x <= 1e-7):
            sols.append(r)
    sols.sort(key=lambda r: r.fun)
    for r in sols[:4]:
        for tol in (1e-6, 1e-4, 1e-8, 1e-3):
            res = attempt(s, L, R, Rn, M, q, r.x, tol)
            if res[0]:
                return res
    return False, None


def attempt(s, L, R, Rn, M, q, x, tol):
    sc = max(1.0, np.abs(x).max())
    F = [k for k in range(len(R)) if abs(Rn[k] @ x[:4]) <= tol * sc]
    v0 = sp.Matrix([rat(a) for a in q - 1 / (1 + np.exp(-(M @ x)))])
    # exact linear conditions on v: sum pi v = 0, and s L^T (pi v) in the row space of R_F (orthogonal to null(R_F))
    Wd = sp.diag(*PI)
    cons = [sp.Matrix([PI]).reshape(1, n)]
    RF = sp.Matrix([R[k] for k in F]) if F else sp.zeros(0, 4)
    null = RF.nullspace() if F else [sp.eye(4)[:, c] for c in range(4)]
    for nv in null:
        cons.append(nv.T * (s * L.T * Wd))
    C = sp.Matrix.vstack(*cons)
    v = v0 - C.T * (C * C.T).pinv() * (C * v0)
    if C * v != sp.zeros(C.rows, 1):
        return False, None
    if not base.in_cone(list(s * L.T * Wd * v), R):
        return False, None
    sv = [qiv[i] - ivr(v[i]) for i in range(n)]
    if not all(si.a > 0 and si.b < 1 for si in sv):
        return False, None
    lb = sum((ivr(PI[i]) * (H(sv[i]) - H(qiv[i])) for i in range(n)), iv.mpf(0))
    return lb.a > UB, lb.a


cs = base.cells()
lbs, fails = [], []
for s in (1, -1):
    for c in cs:
        ok, lb = certify_kl(s, c)
        (lbs if ok else fails).append(lb if ok else (s, c[0], sorted(c[1]), None if lb is None else float(lb)))
print(f"pieces: {2 * len(cs)}; certified above the part-1 value: {len(lbs)}; not certified: {len(fails)}")
if lbs:
    print("least certified lower bound on KL:", float(min(lbs)))
for f_ in fails[:8]:
    print("  not certified:", f_)
ok = not fails
json.dump({"UB": float(UB), "min_lb": float(min(lbs)) if lbs else None, "fails": [str(f_) for f_ in fails[:50]],
           "all_ok": ok}, open(os.path.join(os.path.dirname(__file__), "t34_kl.json"), "w"), indent=1)
print("ALL OK:", ok)
