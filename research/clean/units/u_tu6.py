# serves: TU6 (UNIT-BRANCH.md)
"""Proposition TU6 on T61's instance (D40, kappa = 2, m = 2), exact fractions where a claim is exact.
1. joint build at budgets ({A,C}: 1, {B,D}: 2) is exact on P (offset carried by a constant unit);
2. no covering unit of {A,C} is on at both A and C (part 4);
3. a numeric search for D2*({A,C}, 1) (free class (b)), to see the lower bound 1/8 is respected (not a proof).
"""
import itertools
import json
import os
from fractions import Fraction as Fr

import numpy as np
from scipy.optimize import minimize


def circle_pt(i):
    return (Fr(1 - i * i, 1 + i * i), Fr(2 * i, 1 + i * i))


xs = [circle_pt(1), circle_pt(2)]
eta = 1 - max(a[0] * b[0] + a[1] * b[1] for a, b in itertools.permutations(xs, 2))
r = 1 - eta / 8
c = 2 / (1 - r)
units = {"A": (0, 0), "B": (0, 1), "C": (1, 0), "D": (1, 1)}


def vertex(u):
    j, i = units[u]
    x = [Fr(0)] * 4
    x[2 * j], x[2 * j + 1] = xs[i]
    return x


def z(u, x):
    j, i = units[u]
    return xs[i][0] * x[2 * j] + xs[i][1] * x[2 * j + 1] - r


def relu(t):
    return t if t > 0 else Fr(0)


bad = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D")]
pts = [(u, vertex(u)) for u in "ABCD"] + [("mid" + a + b, [(p + q) / 2 for p, q in zip(vertex(a), vertex(b))]) for a, b in bad]
out = {}

# 1. joint build, offset 0 kept, constant unit relu(1) written with -18 in the {B,D} box
ok = True
for name, x in pts:
    gN = c * sum(relu(z(u, x)) for u in "ABCD")
    gB = relu(-20 * z("A", x) - 40 * z("C", x) - Fr(45, 2)) + relu(-40 * z("B", x) - 20 * z("D", x) - Fr(45, 2)) - 18 * relu(Fr(1))
    ok &= gB == gN
out["joint_exact"] = ok

# 2. covering impossibility: z_A + z_C at A and at C
zA_A, zC_A, zA_C, zC_C = z("A", vertex("A")), z("C", vertex("A")), z("A", vertex("C")), z("C", vertex("C"))
out["zA(A), zC(A), zA(C), zC(C)"] = [str(v) for v in (zA_A, zC_A, zA_C, zC_C)]
out["r>1/2"] = r > Fr(1, 2)
out["B,D off on A,midAC,C"] = all(z(u, x) < 0 for u in "BD" for n, x in pts if n in ("A", "C", "midAC"))

# 3. numeric D2*({A,C},1): gap = w relu(mA zA + mC zC + beta) + wB relu(zB) + wD relu(zD), offset 0
Z = np.array([[float(z(u, x)) for u in "ABCD"] for _, x in pts])
g = np.array([float(c * sum(relu(z(u, x)) for u in "ABCD")) for _, x in pts])


def loss(th):
    mA, mC, beta, w, wB, wD = th
    gb = w * np.maximum(mA * Z[:, 0] + mC * Z[:, 2] + beta, 0) + wB * np.maximum(Z[:, 1], 0) + wD * np.maximum(Z[:, 3], 0)
    return np.mean((g - gb) ** 2)


rng = np.random.default_rng(0)
best = min((minimize(loss, rng.normal(size=6) * s, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-10, fatol=1e-14))
            for s in (1, 10, 50) for _ in range(200)), key=lambda o: o.fun)
out["numeric_D2star_AC_1"] = best.fun
out["bound_1_over_8_respected"] = bool(best.fun >= 1 / 8 - 1e-9)
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "u_tu6.json"), "w"), indent=1)
