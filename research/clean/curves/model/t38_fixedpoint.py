# serves: T38
"""Exact check of Theorem T38 (COARSE-THEORY.md B.O) on the explicit least builds of T20, T32, T33, T34 (D2).

For each build: residual r = g - Delta_N on the points of P (exact rationals). Check
  (1) sum_i pi_i r_i = 0;
  (2) for every unit j: some lambda in [0, 1]^{K_j} (kink points) with
      sum_{A_j} pi r x~ + sum_{K_j} lambda pi r x~ = 0.
(2) is decided exactly: the solution set of the linear equations in lambda is found with sympy, and a point of it in
[0, 1]^{K_j} is checked exactly (a vertex search over the free parameters when the solution is not unique).
Prediction (written before the run): every build satisfies (1) and (2).
"""
import itertools
import json
import os

import sympy as sp

R = sp.Rational
relu = lambda v: v if v > 0 else 0


def orbit3(p):
    out, q = [], tuple(p)
    for _ in range(4):
        out.append(q); q = (-q[1], q[0], q[2])
    return out


def check(name, pts, pi, target, units, gamma):
    """units: list of (sign, theta) with theta = (w..., b); pts: list of tuples."""
    xt = [list(p) + [1] for p in pts]
    pre = [[sum(th[c] * xt[i][c] for c in range(len(th))) for i in range(len(pts))] for _, th in units]
    g = [sum(s * relu(pre[j][i]) for j, (s, _) in enumerate(units)) + gamma for i in range(len(pts))]
    r = [g[i] - target[i] for i in range(len(pts))]
    ok1 = sum(pi[i] * r[i] for i in range(len(pts))) == 0
    ok2 = True
    for j in range(len(units)):
        A = [i for i in range(len(pts)) if pre[j][i] > 0]
        K = [i for i in range(len(pts)) if pre[j][i] == 0]
        base = [sum(pi[i] * r[i] * xt[i][c] for i in A) for c in range(len(xt[0]))]
        lam = sp.symbols(f"l0:{max(1, len(K))}")[:len(K)]
        eqs = [base[c] + sum(lam[a] * pi[i] * r[i] * xt[i][c] for a, i in enumerate(K)) for c in range(len(xt[0]))]
        if not K:
            ok2 = ok2 and all(e == 0 for e in eqs)
            continue
        sols = sp.linsolve(eqs, lam)
        found = False
        for sol in sols:
            free = sorted(set().union(*[e.free_symbols for e in sol]), key=str)
            for vals in itertools.product([0, R(1, 2), 1], repeat=len(free)):
                sub = dict(zip(free, vals))
                v = [e.subs(sub) for e in sol]
                if all(0 <= x <= 1 for x in v):
                    found = True
                    break
            if found:
                break
        ok2 = ok2 and found
    d2 = sum(pi[i] * r[i] ** 2 for i in range(len(pts))) / 4
    print(f"{name}: D2 = {d2}; mean residual zero: {ok1}; every unit's region condition: {ok2}")
    return ok1 and ok2


results = {}
# T20: points -3..3, gap T, least D2 build relu(x - 8/5) + 3/5
P20 = [(x,) for x in range(-3, 4)]
T20 = [max(x - 1, 0) + max(-x - 1, 0) for (x,) in P20]
results["T20"] = check("T20 width 1", P20, [R(1, 7)] * 7, T20, [(1, (1, R(-8, 5)))], R(3, 5))
# T32: square family, eight points, width 2
P32 = [(2, 0), (0, 2), (-2, 0), (0, -2), (2, 2), (-2, 2), (-2, -2), (2, -2)]
T32 = [sum(relu(w[0] * a + w[1] * b - 1) for w in [(1, 0), (0, 1), (-1, 0), (0, -1)]) for a, b in P32]
results["T32"] = check("T32 width 2", P32, [R(1, 8)] * 8, T32,
                       [(1, (R(-4, 5), R(4, 5), R(-8, 5))), (-1, (R(-3, 10), R(3, 10), 0))], R(8, 5))
# T33: leaning square family, three orbits, weights 3, 1, 2
P33 = sum((orbit3(s) for s in [(-1, 2, -2), (2, 1, -1), (1, 0, 2)]), [])
W33 = [(1, 0, R(1, 2)), (0, 1, R(1, 2)), (-1, 0, R(1, 2)), (0, -1, R(1, 2))]
T33 = [sum(relu(w[0] * a + w[1] * b + w[2] * c - 1) for w in W33) for a, b, c in P33]
pi33 = [R(w, 24) for w in (3, 1, 2) for _ in range(4)]
results["T33"] = check("T33 width 1", P33, pi33, T33, [(-1, (0, 0, R(-1, 2), 0))], 1)
# T34: two families, three orbits, uniform
P34 = sum((orbit3(s) for s in [(2, 0, 2), (-1, 2, 0), (1, -1, 0)]), [])
W34 = W33 + [(1, 1, R(-1, 2)), (-1, 1, R(-1, 2)), (-1, -1, R(-1, 2)), (1, -1, R(-1, 2))]
T34 = [sum(relu(w[0] * a + w[1] * b + w[2] * c - 1) for w in W34) for a, b, c in P34]
results["T34"] = check("T34 width 1", P34, [R(1, 12)] * 12, T34,
                       [(1, (R(-11, 3), R(11, 9), R(-10, 9), R(-44, 9)))], R(16, 9))
ALL = all(results.values())
print("ALL OK:", ALL)
json.dump({k: bool(v) for k, v in results.items()} | {"all_ok": bool(ALL)},
          open(os.path.join(os.path.dirname(__file__), "t38_fixedpoint.json"), "w"), indent=1)
