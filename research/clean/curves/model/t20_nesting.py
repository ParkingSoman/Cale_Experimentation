# serves: T20
"""Exact and certified check of Theorem T20 (COARSE-THEORY.md B.T).

Instance: P uniform on x in {-3, ..., 3}, teacher gap T(x) = relu(x - 1) + relu(-x - 1), n_y = 2.
Part 1 (exact, sympy): every width-2 build with gap equal to T on all seven points. A unit is relu(s (x - t)) up to a
positive scale (s = +1 or -1), or a constant. The kink t is either one of the seven points or lies in one of the eight
open intervals between them (or outside). On each such piece the data vector is affine in t, so with p = beta s and
q = beta s t the equation "gap = T" is linear in (p_1, q_1, p_2, q_2, gamma). We solve it exactly on every pair of
pieces and keep the solutions with t inside the piece.
Part 2a (exact, D2): least-squares values for the four units, and the build relu(x - 8/5) + 3/5.
Part 2b (certified, KL): for each of the four units a dual lower bound (1/7) sum (H(s_i) - H(q_i)) at an s that meets
the two linear constraints exactly (enclosed with interval arithmetic), against an interval upper bound on D_P for an
explicit rational build with unit relu(x - 3/2).
Prediction (written before the run): part 1 finds exactly the two unit pairs of T20 part 1; part 2a gives 85/26,
493/110 and 16/5; part 2b certifies every lower bound above the competitor's upper bound.
"""
import itertools
import json
import os
from fractions import Fraction

import numpy as np
import sympy as sp
from mpmath import iv, mp
from scipy.optimize import minimize

XS = list(range(-3, 4))
T = [max(x - 1, 0) + max(-x - 1, 0) for x in XS]
OUT = {}

# ---------------- Part 1: all exact width-2 fits ----------------
pts = XS
pieces = [("pt", Fraction(m)) for m in pts]
bounds = [-sp.oo] + pts + [sp.oo]
pieces += [("iv", (bounds[i], bounds[i + 1])) for i in range(len(bounds) - 1)]


def unit_vec(piece, s, p, q):
    """beta * relu(s (x - t)) on the points, with p = beta s, q = beta s t (affine in (p, q) on the piece)."""
    kind, val = piece
    out = []
    for x in XS:
        if kind == "pt":
            t = val
            out.append(p * (x - t) if s * (x - t) > 0 else 0)
        else:
            a, b = val
            # active when s (x - t) > 0 for t in (a, b): s = +1 needs x >= b, s = -1 needs x <= a
            active = (x >= b) if s == 1 else (x <= a)
            out.append(p * x - q if active else 0)
    return out


sols = []
p1, q1, p2, q2, g = sp.symbols("p1 q1 p2 q2 g")
for (pa, pb) in itertools.combinations_with_replacement(range(len(pieces)), 2):
    for s1, s2 in itertools.product([1, -1], repeat=2):
        A, B = pieces[pa], pieces[pb]
        v1 = unit_vec(A, s1, p1, q1)
        v2 = unit_vec(B, s2, p2, q2)
        eqs = [sp.nsimplify(v1[i] + v2[i] + g - T[i]) for i in range(7)]
        unknowns = [p1, p2, g] + ([q1] if A[0] == "iv" else []) + ([q2] if B[0] == "iv" else [])
        res = sp.linsolve(eqs, unknowns)
        for sol in res:
            d = dict(zip(unknowns, sol))
            if any(e.free_symbols for e in sol):
                raise SystemExit(f"family of solutions on pieces {A},{B}: {sol} -- treat by hand")
            ok = True
            ts = []
            for piece, pp, qq in [(A, d[p1], d.get(q1)), (B, d[p2], d.get(q2))]:
                if pp == 0:
                    ok = False  # a zero unit: the build is really of width 1
                    break
                if piece[0] == "pt":
                    ts.append(sp.Rational(piece[1].numerator, piece[1].denominator))
                else:
                    t = qq / pp
                    a, b = piece[1]
                    if not (a < t < b):
                        ok = False
                        break
                    ts.append(t)
            if ok:
                sols.append({"units": [(s1, str(ts[0]), str(d[p1])), (s2, str(ts[1]), str(d[p2]))],
                             "gamma": str(d[g])})
# the output weight beta = p s may have either sign; the unit is fixed by (s, t)
uniq = sorted({(tuple(sorted((u[0], u[1]) for u in so["units"])), so["gamma"]) for so in sols})
OUT["part1_exact_fits"] = [{"units(sign,kink)": list(u), "gamma": gm} for u, gm in uniq]
print("Part 1: exact width-2 fits (sign s, kink t) -> unit relu(s (x - t)):")
for u, gm in uniq:
    print("  ", u, "gamma", gm)
expected = {(((-1, "-1"), (1, "1")), "0"), (((-1, "1"), (1, "-1")), "-2")}
part1_ok = set(uniq) == expected
OUT["part1_ok"] = part1_ok
print("  matches T20 part 1:", part1_ok)

# ---------------- Part 2a: D2 exact ----------------
def relu(v):
    return max(v, 0)


def ls_value(u):
    a, gg = sp.symbols("a gg")
    S = sum((a * ui + gg - Ti) ** 2 for ui, Ti in zip(u, T))
    sol = sp.solve([sp.diff(S, a), sp.diff(S, gg)], [a, gg], dict=True)[0]
    return sp.nsimplify(S.subs(sol)), sol


four = {
    "relu(x-1)": [relu(x - 1) for x in XS],
    "relu(-x-1)": [relu(-x - 1) for x in XS],
    "relu(x+1)": [relu(x + 1) for x in XS],
    "relu(1-x)": [relu(1 - x) for x in XS],
}
d2 = {}
for name, u in four.items():
    v, sol = ls_value(u)
    d2[name] = v
    print(f"Part 2a: D2 residual for {name}: {v} at {sol}")
comp = sum((relu(x - sp.Rational(8, 5)) + sp.Rational(3, 5) - Ti) ** 2 for x, Ti in zip(XS, T))
print("Part 2a: residual of relu(x - 8/5) + 3/5:", comp)
part2a_ok = (d2["relu(x-1)"] == sp.Rational(85, 26) and d2["relu(x+1)"] == sp.Rational(493, 110)
             and comp == sp.Rational(16, 5) and all(comp < v for v in d2.values()))
OUT["part2a"] = {k: str(v) for k, v in d2.items()} | {"competitor": str(comp), "ok": part2a_ok}

# ---------------- Part 2b: KL certified ----------------
mp.dps = 60
iv.dps = 60
sig = lambda t: 1 / (1 + np.exp(-t))


def kl_np(f):
    q = sig(np.array(T, float))
    s = sig(f)
    return np.mean(q * np.log(q / s) + (1 - q) * np.log((1 - q) / (1 - s)))


def ivH(s):
    return -s * iv.log(s) - (1 - s) * iv.log(1 - s)


def ivsig(t):
    return 1 / (1 + iv.exp(-t))


qiv = [ivsig(iv.mpf(Ti)) for Ti in T]
lower = {}
S0 = {}
for name, u in four.items():
    U = np.array(u, float)
    r = minimize(lambda p: kl_np(p[0] * U + p[1]), [0.5, 0.5], method="BFGS", options={"gtol": 1e-12})
    s0 = [Fraction(float(sig(r.x[0] * ui + r.x[1]))).limit_denominator(10 ** 12) for ui in u]
    # exact feasible s* = s0 + A z*, with (A^T A) z* = A^T (q - s0); enclose z* in interval arithmetic
    A = [[iv.mpf(ui), iv.mpf(1)] for ui in u]
    rhs = [sum(A[i][c] * (qiv[i] - iv.mpf(s0[i].numerator) / s0[i].denominator) for i in range(7)) for c in range(2)]
    G = [[sum(A[i][a] * A[i][b] for i in range(7)) for b in range(2)] for a in range(2)]
    det = G[0][0] * G[1][1] - G[0][1] * G[1][0]
    z = [(G[1][1] * rhs[0] - G[0][1] * rhs[1]) / det, (G[0][0] * rhs[1] - G[1][0] * rhs[0]) / det]
    sstar = [iv.mpf(s0[i].numerator) / s0[i].denominator + A[i][0] * z[0] + A[i][1] * z[1] for i in range(7)]
    assert all(si.a > 0 and si.b < 1 for si in sstar)
    lb = sum(ivH(sstar[i]) - ivH(qiv[i]) for i in range(7)) / 7
    lower[name] = lb
    S0[name] = [str(v) for v in s0]
    print(f"Part 2b: KL inf for {name}: numeric {r.fun:.10f}, certified lower bound in [{lb.a}, {lb.b}]"[:200])

# competitor: unit relu(x - 3/2), best (alpha, gamma) rounded to rationals
U = np.array([relu(x - 1.5) for x in XS], float)
r = minimize(lambda p: kl_np(p[0] * U + p[1]), [0.5, 0.5], method="BFGS", options={"gtol": 1e-12})
# the printed competitor (COARSE-THEORY.md, T20 proof part 2), evaluated directly; r.x is only a cross-check
al, ga = Fraction(969140, 968833), Fraction(219763, 439837)
print("   numerical optimum for the competitor's head:", r.x, "printed:", float(al), float(ga))
ivq = lambda fr: iv.mpf(fr.numerator) / fr.denominator
fc = [ivq(al) * ivq(relu(Fraction(x) - Fraction(3, 2))) + ivq(ga) for x in XS]
up = sum(qiv[i] * iv.log(qiv[i] / ivsig(fc[i])) + (1 - qiv[i]) * iv.log((1 - qiv[i]) / (1 - ivsig(fc[i])))
         for i in range(7)) / 7
print(f"Part 2b: competitor relu(x - 3/2): alpha {al}, gamma {ga}, D_P in [{up.a}, {up.b}]"[:200])
part2b_ok = all(lower[n].a > up.b for n in four)
# the rational bounds printed in the paper (T20 proof, part 2)
for nm, bound in (("relu(x-1)", 416449), ("relu(-x-1)", 416449), ("relu(x+1)", 583137), ("relu(1-x)", 583137)):
    part2b_ok = part2b_ok and bool(lower[nm].a > iv.mpf(bound) / 10 ** 7)
part2b_ok = part2b_ok and bool(up.b < iv.mpf(410684) / 10 ** 7)
margins = {n: float(mp.mpf(lower[n].a) - mp.mpf(up.b)) for n in four}
print("Part 2b: certified margins (lower bound minus competitor upper bound):", margins)
OUT["part2b"] = {"competitor": {"alpha": str(al), "gamma": str(ga), "upper": float(mp.mpf(up.b))},
                 "lower": {n: float(mp.mpf(lower[n].a)) for n in four}, "margins": margins, "s0": S0, "ok": part2b_ok}

ALL = part1_ok and part2a_ok and part2b_ok
OUT["all_ok"] = ALL
print("ALL OK:", ALL)
json.dump(OUT, open(os.path.join(os.path.dirname(__file__), "t20_nesting.json"), "w"), indent=1, default=str)
