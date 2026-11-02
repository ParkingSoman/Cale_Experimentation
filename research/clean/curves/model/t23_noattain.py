# serves: T23
"""Exact check of Theorem T23 (COARSE-THEORY.md B.T): the least need not exist.

Points s_i = (i - 3, 0), i = 1..5, s_6 = (1, 1). Target gap pair (c1, c2), c1 = (2,1,0,-2,-4,0), c2 = (0,0,0,0,0,1).
1. The teacher of Lemma T22a with logits (c1, c2, 0) is exact on all six points (rationals).
2. The sequence of width-2 builds of the proof: outputs equal (c1, relu(x)/m) and (0, 1) exactly (sympy, symbolic m);
   the KL and D2 objectives at m = 10, 100, 1000 decrease toward 0.
3. Not reached: every relu data vector u = a relu(sgn (x - t)) on the five collinear points (kink at each point or in
   each open interval, both signs, a > 0) that equals alpha + beta c1 there has beta = 0 (sympy, exact, piece by
   piece). Constant units give beta = 0 directly.
Prediction (written before the run): 1 and 3 hold exactly; in 2 the objectives fall toward 0 (like 1/m^2).
"""
import itertools
import json
import os
from fractions import Fraction as Fr

import numpy as np
import sympy as sp

S = [(i - 3, 0) for i in range(1, 6)] + [(1, 1)]
c1 = [2, 1, 0, -2, -4, 0]
c2 = [0, 0, 0, 0, 0, 1]
out = {}

# 1. teacher (Lemma T22a, vector labels)
pts = [tuple(Fr(v) for v in s) for s in S]
n = len(pts)
for tau in range(1, n * n + 2):
    v = [Fr(1), Fr(tau)]
    sv = [v[0] * p[0] + v[1] * p[1] for p in pts]
    if len(set(sv)) == n:
        break
order = sorted(range(n), key=lambda i: sv[i])
ss = [sv[i] for i in order]


def broken(labels):
    tt = [Fr(labels[i]) for i in order]
    mu = [Fr(0)] + [(tt[j + 1] - tt[j]) / (ss[j + 1] - ss[j]) for j in range(n - 1)]
    coef = [mu[j + 1] - mu[j] for j in range(n - 1)]
    return lambda z: tt[0] + sum(c * max(z - ss[j], Fr(0)) for j, c in enumerate(coef))


g1, g2 = broken(c1), broken(c2)
teach_ok = all(g1(sv[i]) == c1[i] and g2(sv[i]) == c2[i] for i in range(n))
out["teacher_exact"] = teach_ok
print("1. teacher (width 5) exact on all six points:", teach_ok)

# 2. the sequence
m = sp.symbols("m", positive=True)
W = sp.Matrix([[-1, 2 * m + 1], [1, m - 1]])
A = sp.Matrix([[1, -2], [0, 1 / m]])
outs = []
for s in S:
    h = (W * sp.Matrix(s)).applyfunc(lambda e: sp.Max(e, 0))
    outs.append(sp.simplify(A * h))
seq_ok = True
for i, (s, o) in enumerate(zip(S, outs)):
    x = s[0]
    want1 = sp.Max(x, 0) / m if i < 5 else 1
    seq_ok = seq_ok and sp.simplify(o[0] - c1[i]) == 0 and sp.simplify(o[1] - want1) == 0
print("2. outputs of the m-th build equal (c1, relu(x)/m) and (0, 1) exactly:", seq_ok)


def objectives(mv):
    Y = np.array([[float(o[0].subs(m, mv)), float(o[1].subs(m, mv)), 0.0] for o in outs])
    Yn = np.array([[c1[i], c2[i], 0.0] for i in range(6)])
    p = np.exp(Yn) / np.exp(Yn).sum(1, keepdims=True)
    q = np.exp(Y) / np.exp(Y).sum(1, keepdims=True)
    kl = float(np.mean(np.sum(p * np.log(p / q), 1)))
    D = Y - Yn
    D = D - D.mean(1, keepdims=True)
    d2 = float(0.5 * np.mean(np.sum(D ** 2, 1)))
    return kl, d2


vals = {mv: objectives(mv) for mv in (10, 100, 1000)}
for mv, (kl, d2) in vals.items():
    print(f"   m = {mv}: D_P = {kl:.3e}, D2_P = {d2:.3e}")
dec_ok = vals[10][0] > vals[100][0] > vals[1000][0] and vals[1000][0] < 1e-5
out["sequence_exact"] = bool(seq_ok)
out["objectives"] = {str(k): v for k, v in vals.items()}

# 3. relu data vectors in U on the five collinear points
xs = [-2, -1, 0, 1, 2]
al, be, a, qv = sp.symbols("al be a qv")
bad = []
pieces = [("pt", x0) for x0 in xs] + [("iv", (lo, hi)) for lo, hi in zip([-sp.oo] + xs, xs + [sp.oo])]
for (kind, val), sgn in itertools.product(pieces, (1, -1)):
    # unit a * relu(sgn (x - t)), a > 0; on an open interval the active values are a sgn x - qv, qv = a sgn t
    vec = []
    for x in xs:
        if kind == "pt":
            vec.append(a * sgn * (x - val) if sgn * (x - val) > 0 else 0)
        else:
            lo, hi = val
            active = (x >= hi) if sgn == 1 else (x <= lo)
            vec.append(a * sgn * x - qv if active else 0)
    eqs = [vec[i] - (al + be * c1[i]) for i in range(5)]  # c2 is 0 on the five points
    unknowns = [a, al, be] + ([qv] if kind == "iv" else [])
    for sol in sp.linsolve(eqs, unknowns):
        d = dict(zip(unknowns, sol))
        if any(e.free_symbols for e in sol):
            # a family of solutions: record it, it needs a hand check
            if d[be] != 0:
                bad.append((kind, str(val), sgn, "family", str(sol)))
            continue
        if d[be] != 0 and d[a] > 0:
            if kind == "pt":
                bad.append((kind, str(val), sgn, str(sol)))
            else:
                tval = d[qv] / (d[a] * sgn)
                if val[0] < tval < val[1]:
                    bad.append((kind, str(val), sgn, str(sol)))
noreach_ok = len(bad) == 0
print("3. relu data vectors in U with beta != 0:", bad if bad else "none")
out["not_reached_exact"] = noreach_ok
ALL = teach_ok and seq_ok and dec_ok and noreach_ok
out["all_ok"] = bool(ALL)
print("ALL OK:", ALL)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "t23_noattain.json"), "w"), indent=1, default=str)
