# serves: T17
# serves: T17.2
"""Symbolic checks for Part B.7 of COARSE-THEORY.md (Theorem T17, Theorem T17.2, Proposition T17.3).

S1  (T17 part 1) With a = Gamma w + e, e independent of w with E e = 0 and E e e^T = C (symmetric):
    E[(alpha s^T a + beta) M a | w] - [(alpha s^T Gamma w + beta) M Gamma w + alpha M C s] = 0.
    d = 3, k = 2, n_y = 2, every matrix symbolic.
S2  (T17.2 part 1, T17.3 part 1) For w ~ N(0, I_k), the trace of Cov((alpha b^T w + beta) C w) equals
    alpha^2 (|b|^2 |C|_F^2 + |C b|^2) + beta^2 |C|_F^2. k = 3, C 2 x 3, every entry symbolic (Isserlis moments).
S3  (T17.2 part 3) With mu D = A + B: mu D (2 alpha^2 A^2 + beta^2 D) - B (4 alpha^2 A^2 + beta^2 D)
    = A (2 alpha^2 A (A - B) + beta^2 D).
S4  (T17.2 part 3) n = 2 eigenvalues: g(mu) prod (mu - lambda_j)^3 has degree 3n - 2 = 4 with leading coefficient
    2 alpha^2 R^2 + beta^2 R, R = r_1 + r_2; for beta = 0, (A - B) prod (mu - lambda_j)^2 has degree 2n - 1 = 3 and
    leading coefficient R.
"""
import itertools
import json
import os
import sympy as sp

out = {}
al, be = sp.symbols("alpha beta", real=True)

# S1
d, k, ny = 3, 2, 2
w = sp.Matrix(sp.symbols(f"w0:{k}", real=True))
e = sp.Matrix(sp.symbols(f"e0:{d}", real=True))
G = sp.Matrix(d, k, lambda i, j: sp.Symbol(f"G{i}{j}", real=True))
s = sp.Matrix(sp.symbols(f"s0:{d}", real=True))
M = sp.Matrix(ny, d, lambda i, j: sp.Symbol(f"M{i}{j}", real=True))
C = sp.Matrix(d, d, lambda i, j: sp.Symbol(f"C{min(i, j)}{max(i, j)}", real=True))
a = G * w + e
expr = ((al * (s.T * a)[0] + be) * (M * a)).applyfunc(sp.expand)


def cond_mean_e(poly):
    """E over e (mean 0, second moment C) of a polynomial of degree <= 2 in e, with w fixed."""
    P = sp.Poly(poly, *e)
    res = 0
    for mon, coef in P.terms():
        deg = sum(mon)
        if deg == 0:
            res += coef
        elif deg == 1:
            pass
        elif deg == 2:
            idx = [i for i, m in enumerate(mon) for _ in range(m)]
            res += coef * C[idx[0], idx[1]]
        else:
            raise ValueError("degree > 2 in e")
    return sp.expand(res)


lhs = expr.applyfunc(cond_mean_e)
rhs = ((al * (s.T * G * w)[0] + be) * (M * G * w) + al * M * C * s).applyfunc(sp.expand)
out["S1 E[gate x value | w] - formula"] = str((lhs - rhs).applyfunc(sp.simplify))

# S2
k2, n2 = 3, 2
w2 = sp.symbols(f"x0:{k2}", real=True)
b = sp.Matrix(sp.symbols(f"b0:{k2}", real=True))
Cm = sp.Matrix(n2, k2, lambda i, j: sp.Symbol(f"c{i}{j}", real=True))
wv = sp.Matrix(w2)
X = ((al * (b.T * wv)[0] + be) * (Cm * wv)).applyfunc(sp.expand)


def gauss_E(poly):
    """E over independent standard normals x0..x{k-1}: E x^n = (n-1)!! for even n, 0 for odd n."""
    P = sp.Poly(sp.expand(poly), *w2)
    res = 0
    for mon, coef in P.terms():
        m = 1
        for p in mon:
            m *= 0 if p % 2 else sp.factorial2(p - 1)
        res += coef * m
    return sp.expand(res)


second = sum(gauss_E(X[i] ** 2) for i in range(n2))
first = sum(gauss_E(X[i]) ** 2 for i in range(n2))
trcov = sp.expand(second - first)
CF2 = sum(c ** 2 for c in Cm)
formula = sp.expand(al ** 2 * ((b.T * b)[0] * CF2 + ((Cm * b).T * (Cm * b))[0]) + be ** 2 * CF2)
out["S2 tr Cov - formula"] = str(sp.simplify(trcov - formula))

# S3
A, B, D = sp.symbols("A B D", real=True)
lhs3 = (A + B) * (2 * al ** 2 * A ** 2 + be ** 2 * D) - B * (4 * al ** 2 * A ** 2 + be ** 2 * D)
rhs3 = A * (2 * al ** 2 * A * (A - B) + be ** 2 * D)
out["S3 cancellation"] = str(sp.expand(lhs3 - rhs3))

# S4
mu, l1, l2, r1, r2 = sp.symbols("mu l1 l2 r1 r2", real=True)
lams, rs = (l1, l2), (r1, r2)
A_ = sum(r / (mu - l) for r, l in zip(rs, lams))
D_ = sum(r / (mu - l) ** 2 for r, l in zip(rs, lams))
B_ = sum(l * r / (mu - l) ** 2 for r, l in zip(rs, lams))
prod = (mu - l1) * (mu - l2)
gpoly = sp.Poly(sp.cancel((2 * al ** 2 * A_ * (A_ - B_) + be ** 2 * D_) * prod ** 3), mu)
R = r1 + r2
out["S4 degree (expect 4)"] = gpoly.degree()
out["S4 leading - (2 alpha^2 R^2 + beta^2 R)"] = str(sp.simplify(gpoly.LC() - (2 * al ** 2 * R ** 2 + be ** 2 * R)))
hpoly = sp.Poly(sp.cancel((A_ - B_) * prod ** 2), mu)
out["S4 beta=0 degree (expect 3)"] = hpoly.degree()
out["S4 beta=0 leading - R"] = str(sp.simplify(hpoly.LC() - R))
out["S4 beta=0 form - sum r (mu - 2 l)/(mu - l)^2"] = str(sp.simplify(
    (A_ - B_) - sum(r * (mu - 2 * l) / (mu - l) ** 2 for r, l in zip(rs, lams))))

print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "t17_sympy.json"), "w"), indent=1)
