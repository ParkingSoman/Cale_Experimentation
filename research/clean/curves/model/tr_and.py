# serves: T58
"""Theorem T58 (COARSE-THEORY.md B.W): an AND group collapses exactly, checked in exact rationals.

1. Random one-hidden-layer and two-hidden-layer networks with a group S whose members share one weak sign at every
   point of the mix (built from a shared direction u: z_i = w_i^T x + b_i with w_i = u + small noise, points far along
   +u or -u). Check the AND property exactly, then replace S by the factorisation of part 2 (k = rank + 1, the
   positive-row case k = rank, and the two-logit split k = 2) and check that every logit is unchanged at every point.
2. Control: a group that is NOT AND (one point where members have both signs). The same replacement changes the logits
   there, so the premise is needed.
"""
import itertools
import json
import os
import random
from fractions import Fraction as Fr

import sympy as sp

R = random.Random(58)


def rq(a=-3, b=3, den=4):
    return Fr(R.randint(a * den, b * den), den)


def relu(t):
    return t if t > 0 else Fr(0)


def matvec(W, x):
    return [sum(a * b for a, b in zip(row, x)) for row in W]


def factor_rank_plus_one(WS):
    """W_S = U V with V >= 0 and rank(W_S) + 1 rows (proof of T58 part 2)."""
    M = sp.Matrix(WS)
    rref, piv = M.T.rref()  # columns of M.T = rows of M
    basis = [list(M.row(i)) for i in piv]
    r = len(basis)
    c = max([Fr(0)] + [-Fr(str(e)) for row in basis for e in row])
    V = [[Fr(str(e)) + c for e in row] for row in basis] + [[Fr(1)] * M.cols]
    # W_S = U0 R with R = basis; R = E V with E = [I, -c 1]
    U0 = sp.Matrix(WS) * sp.Matrix(basis).T * (sp.Matrix(basis) * sp.Matrix(basis).T).inv()
    E = sp.Matrix.hstack(sp.eye(r), -sp.Matrix([c] * r))
    U = U0 * E
    U = [[Fr(str(sp.nsimplify(e))) for e in U.row(i)] for i in range(U.rows)]
    return U, V


def net1(d, n, s, ny):
    u = [rq() for _ in range(d)]
    while all(a == 0 for a in u):
        u = [rq() for _ in range(d)]
    W = [[ui + Fr(R.randint(-1, 1), 8) for ui in u] if i < s else [rq() for _ in range(d)] for i in range(n)]
    b = [Fr(R.randint(-1, 1), 8) if i < s else rq() for i in range(n)]
    A = [[rq() for _ in range(n)] for _ in range(ny)]
    c = [rq() for _ in range(ny)]
    return W, b, A, c, u


def logits1(W, b, A, c, x, S=None, U=None, V=None):
    z = [p + q for p, q in zip(matvec(W, x), b)]
    h = [relu(t) for t in z]
    y = [sum(A[o][i] * h[i] for i in range(len(h)) if S is None or i not in S) + c[o] for o in range(len(A))]
    if S is not None:
        zS = [z[i] for i in S]
        hnew = [relu(sum(v * t for v, t in zip(row, zS))) for row in V]
        y = [y[o] + sum(U[o][t] * hnew[t] for t in range(len(V))) for o in range(len(A))]
    return y


def and_points(u, d, npts):
    pts = []
    for j in range(npts):
        lam = Fr(R.randint(20, 60), 4) * (1 if j % 2 == 0 else -1)
        pts.append([lam * ui + Fr(R.randint(-2, 2), 8) for ui in u])
    return pts


def is_and(W, b, S, pts):
    for x in pts:
        z = [sum(a * q for a, q in zip(W[i], x)) + b[i] for i in S]
        if not (all(t >= 0 for t in z) or all(t <= 0 for t in z)):
            return False
    return True


def main():
    out = {"cases": []}
    ok = True
    for case in range(40):
        d, n, s, ny = R.choice([2, 3, 4]), R.choice([4, 5, 6]), R.choice([2, 3, 4]), R.choice([2, 3])
        W, b, A, c, u = net1(d, n, s, ny)
        S = list(range(s))
        pts = and_points(u, d, 8)
        if not is_and(W, b, S, pts):
            continue
        WS = [[A[o][i] for i in S] for o in range(ny)]
        U, V = factor_rank_plus_one(WS)
        assert all(e >= 0 for row in V for e in row)
        same = all(logits1(W, b, A, c, x) == logits1(W, b, A, c, x, S, U, V) for x in pts)
        # two-logit split on the gap row
        alpha = [A[0][i] - A[1][i] for i in S]
        Vs = [[max(a, Fr(0)) for a in alpha], [max(-a, Fr(0)) for a in alpha]]
        A2 = [[A[0][i] - A[1][i] for i in range(n)], [Fr(0)] * n]
        c2 = [c[0] - c[1], Fr(0)]
        Us = [[Fr(1), Fr(-1)], [Fr(0), Fr(0)]]
        gap_same = all(logits1(W, b, A2, c2, x)[0] == logits1(W, b, A2, c2, x, S, Us, Vs)[0] for x in pts)
        ok &= same and gap_same
        out["cases"].append(dict(d=d, n=n, s=s, ny=ny, k=len(V), rank=len(V) - 1, exact=same, two_logit_k2_exact=gap_same))
    # positive-row case: k = rank when the row space holds a positive vector
    W, b, A, c, u = net1(3, 5, 3, 2)
    S = [0, 1, 2]
    A[0][0], A[0][1], A[0][2] = Fr(1), Fr(2), Fr(3)        # positive row
    A[1][0], A[1][1], A[1][2] = Fr(-1), Fr(1), Fr(0)
    pts = and_points(u, 3, 8)
    q = [Fr(1), Fr(2), Fr(3)]
    r2 = [Fr(-1), Fr(1), Fr(0)]
    c2 = Fr(1)
    V = [q, [a + c2 * b_ for a, b_ in zip(r2, q)]]          # both rows >= 0
    # rows of W_S: row0 = q, row1 = r2 = V1 - c2 V0
    U = [[Fr(1), Fr(0)], [-c2, Fr(1)]]
    pos_same = is_and(W, b, S, pts) and all(logits1(W, b, A, c, x) == logits1(W, b, A, c, x, S, U, V) for x in pts)
    ok &= pos_same
    out["positive_row_k_equals_rank"] = pos_same
    # depth 2: group in layer 1 of a 2-hidden-layer net; layer 2 reads layer 1, logits read layer 2
    W, b, A1, c1, u = net1(3, 5, 3, 4)   # A1 = layer-2 weights (4 units), c1 = layer-2 biases
    Aout = [[rq() for _ in range(4)] for _ in range(2)]
    cout = [rq(), rq()]
    S = [0, 1, 2]
    pts = and_points(u, 3, 8)

    def logits2(x, U=None, V=None):
        h2 = [relu(t) for t in logits1(W, b, A1, c1, x, S if V else None, U, V)]
        return [sum(Aout[o][i] * h2[i] for i in range(4)) + cout[o] for o in range(2)]
    WS = [[A1[o][i] for i in S] for o in range(4)]
    U, V = factor_rank_plus_one(WS)
    deep_same = is_and(W, b, S, pts) and all(logits2(x) == logits2(x, U, V) for x in pts)
    ok &= deep_same
    out["depth2_exact"] = deep_same
    # control: a mixed point breaks it
    W, b, A, c, u = net1(3, 5, 3, 2)
    S = [0, 1, 2]
    b[0], b[1], b[2] = Fr(1), Fr(-1), Fr(1)                # at x = 0: z = (1, -1, 1), both signs -> not AND
    A[0][0], A[0][1], A[0][2] = Fr(1), Fr(1), Fr(1)
    pts = [[Fr(0)] * 3]
    WS = [[A[o][i] for i in S] for o in range(2)]
    U, V = factor_rank_plus_one(WS)
    ctrl_differs = (not is_and(W, b, S, pts)) and any(logits1(W, b, A, c, x) != logits1(W, b, A, c, x, S, U, V) for x in pts)
    out["control_not_and_differs"] = ctrl_differs
    out["n_cases"] = len(out["cases"])
    out["ALL_OK"] = bool(ok and ctrl_differs and out["n_cases"] >= 20)
    with open(os.path.join(os.path.dirname(__file__), "tr_and.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("cases", out["n_cases"], "positive_row", pos_same, "depth2", deep_same, "control_differs", ctrl_differs,
          "ALL_OK", out["ALL_OK"])


if __name__ == "__main__":
    main()
