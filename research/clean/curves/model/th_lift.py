# serves: TH2
"""Proposition TH2 part 2 (COARSE-THEORY.md B.H): lifts do not nest. Chain network a = relu(x), g = relu(a_1),
f_i = relu(g - i), gap f_1 - 2 f_2 + 2 f_3 = 0, 1, 0, 1 on P = {(1,0), (2,1), (3,0), (4,1)}.
Checks in exact fractions: N's gap; the box relu(x_2) at G_3 is exact; at G_1 and G_2 every build's gap is monotone
in g, and the least D2 over ALL monotone gap vectors (an exact lower bound, by pool-adjacent-violators on both orders)
is positive; a numeric search over builds a relu(mu g + nu) + gamma for D_P confirms a positive least value.
Also: the (r) unit at G_3 that reads the kept unit a_2 is exact (the comment of TH2)."""
import json
import os
from fractions import Fraction as Fr

import numpy as np
from scipy.optimize import minimize


def relu(t):
    return t if t > 0 else 0 * t


P = [(Fr(1), Fr(0)), (Fr(2), Fr(1)), (Fr(3), Fr(0)), (Fr(4), Fr(1))]
w = [Fr(1, 4)] * 4
g = [relu(relu(x1)) for x1, _ in P]
f = [[relu(gi - i) for i in (1, 2, 3)] for gi in g]
gapN = [fi[0] - 2 * fi[1] + 2 * fi[2] for fi in f]
print("g =", [str(v) for v in g], "gap_N =", [str(v) for v in gapN])
ok_gap = gapN == [0, 1, 0, 1]

box3 = [relu(x2) for _, x2 in P]
exact_G3 = box3 == gapN          # gap weight 1, bias 0
r_unit = [relu(relu(x2)) for _, x2 in P]   # z of kept unit a_2 is x_2 (a_2 = relu(x_2), x_2 >= 0 on P)
exact_r = r_unit == gapN


def pava(y, wts):
    """Weighted least-squares non-decreasing fit, exact fractions."""
    blocks = [[yv, wv, 1] for yv, wv in zip(y, wts)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            a, b = blocks[i], blocks[i + 1]
            wsum = a[1] + b[1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / wsum, wsum, a[2] + b[2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for v, _, n in blocks:
        out += [v] * n
    return out


best = None
for sign in (1, -1):
    fit = [sign * v for v in pava([sign * t for t in gapN], w)]
    # D2_P = 1/4 E (Delta_B - Delta_N)^2 for two logits; the bias gamma is inside the monotone class
    val = Fr(1, 4) * sum(wi * (a - b) ** 2 for wi, a, b in zip(w, fit, gapN))
    best = val if best is None else min(best, val)
print("least D2 over monotone gaps (lower bound for G_1, G_2):", best)
lower_ok = best > 0

tN = np.array([float(v) for v in gapN])
gN = np.array([float(v) for v in g])


def DP(th):
    a, mu, nu, gam = th
    d = a * np.maximum(mu * gN + nu, 0) + gam
    pN = 1 / (1 + np.exp(-tN))
    pB = 1 / (1 + np.exp(-d))
    pB = np.clip(pB, 1e-15, 1 - 1e-15)
    return float(np.mean(pN * np.log(pN / pB) + (1 - pN) * np.log((1 - pN) / (1 - pB))))


rng = np.random.default_rng(0)
vals = [minimize(DP, rng.normal(size=4) * 3, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-10,
                                                                                 fatol=1e-14)).fun for _ in range(200)]
print("numeric least D_P over a relu(mu g + nu) + gamma (200 starts):", min(vals))
ALL_OK = bool(ok_gap and exact_G3 and exact_r and lower_ok and min(vals) > 1e-3)
print("ALL_OK", ALL_OK)
with open(os.path.join(os.path.dirname(__file__), "th_lift.json"), "w") as fh:
    json.dump(dict(gapN=[str(v) for v in gapN], exact_G3=exact_G3, exact_r_at_G3=exact_r,
                   least_D2_monotone=str(best), numeric_least_DP_G1=min(vals), ALL_OK=ALL_OK), fh, indent=1)
