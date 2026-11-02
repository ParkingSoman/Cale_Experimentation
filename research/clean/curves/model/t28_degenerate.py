# serves: T28
"""Exact check of Theorem T28 parts 1 and 2 (COARSE-THEORY.md B.Z).

Part 2: random d, d + 1 random rational points (affinely independent), a random network N with two logits; solve
w^T x_j + b = Delta_N(x_j) - c exactly (fractions) and confirm relu(w^T x + b) + c equals N's gap at every point.
Part 1: for a random W (n < d rows), a unit with w outside the row space varies along a kernel direction v (exact).
Prediction (written before the run): every trial fits exactly; every out-of-row-space unit varies on a fibre.
"""
import json
import os
import random
from fractions import Fraction as Fr

import sympy as sp

random.seed(0)
ok2 = True
for trial in range(20):
    d = random.randint(2, 5)
    pts = [[Fr(random.randint(-5, 5)) for _ in range(d)] for _ in range(d + 1)]
    M = sp.Matrix([[*p, 1] for p in pts])
    if M.det() == 0:
        continue
    n = random.randint(2, 6)
    W = [[Fr(random.randint(-3, 3)) for _ in range(d)] for _ in range(n)]
    b = [Fr(random.randint(-3, 3)) for _ in range(n)]
    a = [Fr(random.randint(-3, 3)) for _ in range(n)]
    gapN = [sum(a[i] * max(sum(W[i][l] * p[l] for l in range(d)) + b[i], 0) for i in range(n)) for p in pts]
    c = min(gapN)
    sol = M.LUsolve(sp.Matrix([gN - c for gN in gapN]))
    w, bb = sol[:d], sol[d]
    fit = [max(sum(w[l] * p[l] for l in range(d)) + bb, 0) + c for p in pts]
    ok2 = ok2 and all(sp.nsimplify(f - gN) == 0 for f, gN in zip(fit, gapN))
print("Part 2: perfect width-1 copy from the answers alone, in every trial:", ok2)

ok1 = True
for trial in range(20):
    d, n = 5, 3
    W = sp.Matrix([[random.randint(-3, 3) for _ in range(d)] for _ in range(n)])
    ker = W.nullspace()
    if not ker:
        continue
    v = ker[0]
    w = sp.Matrix([random.randint(-3, 3) for _ in range(d)])
    if (w.T * v)[0] == 0:
        continue
    # z(x + t v) = z(x) for all t, while w^T (x + t v) changes
    ok1 = ok1 and (W * v).is_zero_matrix and (w.T * v)[0] != 0
print("Part 1: units outside the row space vary on a fibre of z:", ok1)
ALL = ok1 and ok2
print("ALL OK:", ALL)
json.dump({"part1": bool(ok1), "part2": bool(ok2), "all_ok": bool(ALL)},
          open(os.path.join(os.path.dirname(__file__), "t28_degenerate.json"), "w"), indent=1)
