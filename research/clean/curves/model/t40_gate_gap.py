# serves: T40
"""Exact check of Theorem T40 and Corollary T39.1 (COARSE-THEORY.md B.G), in rational arithmetic (sympy).

T40 (instance I1, uniform weight 1/13): L_G = least squared misfit on G of one affine function of z = h(x), computed
by exact least squares over all 6 coefficients (no symmetry used), times the weights; then D2 of the explicit build B*.
Predictions (from the proof): L_G = 1/26, D2(B*) = 5/182.
T39.1 (instance I3: the same points, members relu(h_i + 5 g - 5), so each member is linear on G): the closed-form
build relu(a(z) + M (g - 1)) + Delta_0 equals N at every point. Prediction: D2 = 0 exactly.
"""
import json
import os

import sympy as sp

R = sp.Rational


def relu(v):
    return v if v > 0 else sp.Integer(0)


def orbit(p):
    out, q = [], tuple(p)
    for _ in range(4):
        out.append(q)
        q = (-q[1], q[0])
    return out


def h(x1, x2, x3):
    return [relu(sp.Integer(x1)), relu(sp.Integer(x2)), relu(sp.Integer(-x1)), relu(sp.Integer(-x2)),
            relu(sp.Integer(x3))]


PTS = [(a, b, 1) for o in [(2, 0), (2, 2), (1, 1)] for (a, b) in orbit(o)] + [(0, 0, 0)]
Z = [h(*p) for p in PTS]
PI = R(1, 13)


def gap_I1(z):
    return sum(relu(z[i] + 5 * z[4] - 6) for i in range(4))


def gap_I3(z):
    return sum(relu(z[i] + 5 * z[4] - 5) for i in range(4))


def exact_LG(gapf):
    G = [z for z in Z if z[4] > 0]
    A = sp.Matrix([z + [1] for z in G])
    t = sp.Matrix([gapf(z) for z in G])
    # least squares via the normal equations on the column space (A may be rank deficient: g is constant on G)
    P = A * (A.T * A).pinv() * A.T
    F = P * t
    res = t - F
    return PI * (res.T * res)[0] / 4, F


def main():
    out = {}
    LG, F = exact_LG(gap_I1)
    out["T40_L_G"] = str(LG)
    bstar = lambda z: relu(R(19, 14) * (z[0] + z[2]) + R(1, 2) * (z[1] + z[3]) - z[4] - 1) + R(2, 7)
    d2 = sum(PI * (bstar(z) - gap_I1(z)) ** 2 / 4 for z in Z)
    out["T40_D2_Bstar"] = str(d2)
    out["T40_holds"] = bool(LG == R(1, 26) and d2 == R(5, 182) and d2 < LG)

    # T39.1 on I3: ON = all four members (pre-activation h_i >= 0 on G); a(z) = sum_i (h_i + 5 g - 5) + c_0 - Delta_0
    pre_on_G = [[z[i] + 5 * z[4] - 5 for i in range(4)] for z in Z if z[4] > 0]
    lin = all(all(v >= 0 for v in row) for row in pre_on_G)
    delta0 = gap_I3(Z[-1])
    a = lambda z: sum(z[i] + 5 * z[4] - 5 for i in range(4)) - delta0
    offG = [z for z in Z if z[4] == 0]
    M = max([a(z) for z in offG] + [0]) + 1
    b = lambda z: relu(a(z) + M * (z[4] - 1)) + delta0
    d2_I3 = sum(PI * (b(z) - gap_I3(z)) ** 2 / 4 for z in Z)
    a_nonneg = all(a(z) >= 0 for z in Z if z[4] > 0)
    out["T39.1_members_linear_on_G"] = bool(lin)
    out["T39.1_a_nonneg_on_G"] = bool(a_nonneg)
    out["T39.1_M"] = str(M)
    out["T39.1_D2"] = str(d2_I3)
    out["T39.1_holds"] = bool(lin and a_nonneg and d2_I3 == 0)
    for k, v in out.items():
        print(k, v)
    with open(os.path.join(os.path.dirname(__file__), "t40_gate_gap.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
