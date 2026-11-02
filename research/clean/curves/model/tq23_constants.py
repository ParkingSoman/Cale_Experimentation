# serves: TQ23
"""Constants of Lemmas TQ23.a, TQ23.b and Theorem TQ23, checked by sympy and on grids. Each line prints the worst value against its bound; fails counts violations.
Estimator, fixed before the run: (1) A''(e) >= P/19 for P in (0, 1/2], |e| <= 1, on a grid, and the exact worst case by endpoint analysis. (2) A(e) >= P min(e^2, 1) / 38. (3) g(c) <= 1 + c and g(-c) >= 1 - c/3 on [0, 1]. (4) The budget and spread arithmetic at alpha = 1/9. (5) (1 + alpha)(1 + alpha) <= 1 + 3 alpha, and the lower factor collection. (6) The gauge split as a sympy identity. (7) The reduction's gap for |E| from 1 to 10^6.
Prediction, written before the run: 0 fails."""
import math

import numpy as np
import sympy as sp

fails = 0


def check(name, ok, val=""):
    global fails
    fails += not ok
    print(("ok  " if ok else "FAIL"), name, val)


P = np.linspace(1e-6, .5, 2001)[None, :]
e = np.linspace(-1, 1, 4001)[:, None]
D = 1 + P * (np.exp(e) - 1)
App = P * (1 - P) * np.exp(e) / D ** 2
check("A'' / P >= 1/19", (App / P).min() >= 1 / 19, (App / P).min() * 19)
e2 = np.linspace(-30, 30, 12001)[:, None]
D2 = 1 + P * (np.exp(e2) - 1)
A = np.log(D2) - P * e2
ratio = (A / (P * np.minimum(e2 ** 2, 1) / 38 + 1e-300))
mask = np.abs(e2[:, 0]) > 1e-3
check("A >= P min(e^2,1)/38", ratio[mask].min() >= 1 - 1e-9, ratio[mask].min())
c = np.linspace(1e-6, 1, 2001)
g = lambda x: 2 * (np.expm1(x) - x) / x ** 2
check("g(c) <= 1 + c", (g(c) <= 1 + c).all())
# g(-c) >= 1 - c/3 is e^-c >= 1 - c + c^2/2 - c^3/6 (Taylor, remainder c^4 e^-xi / 24 >= 0). Floats lose it near c = 0, so test in high precision.
import mpmath as mp
mp.mp.dps = 60
worst = min(mp.mpf(2) * (mp.exp(-x) - 1 + x) / x ** 2 - (1 - x / 3) for x in (mp.mpf(10) ** (-mp.mpf(i) / 10) for i in range(0, 61)))
check("g(-c) >= 1 - c/3, high precision, c from 1e-6 to 1", worst >= 0, worst)
al = 1 / 9
check("budget (1 + 3 alpha)/4 <= 1/3", (1 + 3 * al) / 4 <= 1 / 3 + 1e-12)
check("38 (10/9) / 3 <= 15", 38 * (10 / 9) / 3 <= 15, 38 * (10 / 9) / 3)
check("15 alpha^2 < 1", 15 * al ** 2 < 1)
check("sqrt 15 <= 4", math.sqrt(15) <= 4)
check("e^(-1/18)/1.01 >= 9/10", math.exp(-1 / 18) / 1.01 >= .9)
check("s[g] >= 1/4", (1 / math.e) / (1 + 1 / math.e) / 1.01 >= .25)
als = np.linspace(0, 1 / 9, 501)
check("upper factors", ((1 + als) ** 2 <= 1 + 3 * als + 1e-12).all())
eta = np.linspace(0, .01, 101)[:, None]
lowf = (1 - 8 * als / 3) * np.exp(-als / 2) * (1 - 6 * eta)
check("lower factors >= 1 - 4 alpha - 6 eta", (lowf >= 1 - 4 * als - 6 * eta - 1e-12).all())
check("(1 - 5 eta)/(1 + eta) >= 1 - 6 eta", ((1 - 5 * eta) / (1 + eta) >= 1 - 6 * eta - 1e-12).all())
f, y, t = sp.symbols("f y t", positive=True)
gap = sp.simplify((f - y) ** 2 - ((1 - t) * f ** 2 - (1 / t - 1) * y ** 2) - (t * f - y) ** 2 / t)
check("gauge split identity", gap == 0, gap)
for E in (1, 2, 3, 7, 10, 100, 10 ** 4, 10 ** 6):
    a = 1 / (64 * E ** 2)
    alpha = a * math.sqrt(E)
    eps = 4 * alpha + 1 / (16 * E)
    check(f"|E| = {E}: eps' <= 1/(8|E|), 3 alpha |E| <= 1/8, alpha <= 1/9", eps <= 1 / (8 * E) + 1e-15 and 3 * alpha * E <= 1 / 8 and alpha <= 1 / 9)
    # worst no-instance value: (1 - eps') (X + 5/8) with X = |E| - K in [0, |E|]
    worst = min((1 - eps) * (X + 5 / 8) - X for X in (0, E))
    check(f"|E| = {E}: no-instance >= X + 3/8", worst >= 3 / 8 - 1e-12, worst)
print("fails", fails)
