# serves: D39
"""Remark A.5.1 (COARSE-THEORY.md): the networks of Part B under the principles Q2 and Q3 of Definition D39.

For each network: its first-layer units (w, b), its readers (logit differences), its families (orbits), and its mix.
Q2 fails for a family when two members are both on at some point of the domain; Q3 fails for a reader when two units
it reads with opposite signs are both on at some point of the domain. Domain: the convex hull of the mix's support
(the smallest convex domain that holds the mix; failure there implies failure on every larger convex domain, R^d
included). A failure is shown by an exact rational witness: a mix point or the midpoint of two mix points, both in
the hull. An "apart" verdict is backed by a linear program on the hull (max t with both pre-activations >= t),
which is a numerical confirmation only; the paper proves the apart cases by hand.
Prediction (written before the run): T20 passes Q2 and Q3; the T22 and T23 broken lines fail Q3; T32, T33, T34 fail Q2
and pass Q3; the detector network of D40 (kappa = 2, m = 3) passes Q2 and Q3.
"""
import itertools
import json
import os
import sys
from fractions import Fraction as Fr

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(__file__))
from t22_reduction import build, circle_pt  # noqa: E402


def z(unit, x):
    w, b = unit
    return sum(a * c for a, c in zip(w, x)) + b


def witness(u, v, pts):
    cands = list(pts) + [tuple((a + c) / 2 for a, c in zip(p, q)) for p, q in itertools.combinations(pts, 2)]
    for x in cands:
        if z(u, x) > 0 and z(v, x) > 0:
            return x
    return None


def lp_overlap(u, v, pts):
    P = np.array([[float(c) for c in p] for p in pts])
    n, d = P.shape
    # variables: lambda (n), t ; maximise t s.t. z_u(P^T lambda) >= t, z_v >= t, sum lambda = 1, lambda >= 0, t <= 1
    A, bvec = [], []
    for (w, b) in (u, v):
        wf = np.array([float(c) for c in w])
        A.append(np.concatenate([-(P @ wf), [1.0]])); bvec.append(float(b))
    res = linprog(np.concatenate([np.zeros(n), [-1.0]]), A_ub=np.array(A), b_ub=np.array(bvec),
                  A_eq=np.concatenate([np.ones(n), [0.0]])[None, :], b_eq=[1.0],
                  bounds=[(0, None)] * n + [(None, 1.0)], method="highs")
    return -res.fun


def broken_line(pts, logit_rows):
    """Lemma T22a with one coefficient list per logit difference: returns units and reader weights."""
    n, d = len(pts), len(pts[0])
    for tau in range(1, (d - 1) * n * n + 2):
        v = [Fr(tau) ** l for l in range(d)]
        s = [sum(a * b for a, b in zip(v, p)) for p in pts]
        if len(set(s)) == n:
            break
    order = sorted(range(n), key=lambda i: s[i])
    ss = [s[i] for i in order]
    units = [(tuple(v), -ss[j]) for j in range(n - 1)]
    readers = []
    for row in logit_rows:
        tt = [row[i] for i in order]
        mu = [Fr(0)] + [(tt[j + 1] - tt[j]) / (ss[j + 1] - ss[j]) for j in range(n - 1)]
        readers.append([mu[j + 1] - mu[j] for j in range(n - 1)])
    return units, readers


def rot_z(p):  # 90 degrees about the x_3 axis (or the plane)
    return (-p[1], p[0]) + tuple(p[2:])


def orbit(p):
    out = [tuple(Fr(c) for c in p)]
    for _ in range(3):
        out.append(rot_z(out[-1]))
    return out


nets = {}
nets["T20"] = dict(units=[((Fr(1),), Fr(-1)), ((Fr(-1),), Fr(-1))], readers=[[Fr(1), Fr(1)]],
                   families=[[0, 1]], pts=[(Fr(x),) for x in range(-3, 4)])
I = build({frozenset(((0, 1), (1, 2)))})
u22, r22 = broken_line(I["pts"], [I["lab"]])
nets["T22 teacher (yes-graph, kappa 2)"] = dict(units=u22, readers=r22, families=[[j] for j in range(len(u22))],
                                                pts=I["pts"])
p23 = [(Fr(i - 3), Fr(0)) for i in range(1, 6)] + [(Fr(1), Fr(1))]
c1 = [Fr(c) for c in (2, 1, 0, -2, -4, 0)]
c2 = [Fr(c) for c in (0, 0, 0, 0, 0, 1)]
u23, r23 = broken_line(p23, [c1, c2])
nets["T23 teacher"] = dict(units=u23, readers=r23, families=[[j] for j in range(len(u23))], pts=p23)
e = [(1, 0), (0, 1), (-1, 0), (0, -1)]
nets["T32"] = dict(units=[(tuple(Fr(c) for c in w), Fr(-1)) for w in e], readers=[[Fr(1)] * 4], families=[[0, 1, 2, 3]],
                   pts=[(Fr(a), Fr(b)) for a, b in [(2, 0), (0, 2), (-2, 0), (0, -2), (2, 2), (-2, 2), (-2, -2), (2, -2)]])
wA = [(1, 0, Fr(1, 2)), (0, 1, Fr(1, 2)), (-1, 0, Fr(1, 2)), (0, -1, Fr(1, 2))]
nets["T33"] = dict(units=[(tuple(Fr(c) for c in w), Fr(-1)) for w in wA], readers=[[Fr(1)] * 4],
                   families=[[0, 1, 2, 3]], pts=orbit((-1, 2, -2)) + orbit((2, 1, -1)) + orbit((1, 0, 2)))
wB = [(1, 1, Fr(-1, 2)), (-1, 1, Fr(-1, 2)), (-1, -1, Fr(-1, 2)), (1, -1, Fr(-1, 2))]
nets["T34"] = dict(units=[(tuple(Fr(c) for c in w), Fr(-1)) for w in wA + wB], readers=[[Fr(1)] * 8],
                   families=[[0, 1, 2, 3], [4, 5, 6, 7]], pts=orbit((2, 0, 2)) + orbit((-1, 2, 0)) + orbit((1, -1, 0)))

# Definition D40, kappa = 2, m = 3 (yes-instance mix of Theorem T43 for H = path 1-2-3)
kappa, m = 2, 3
xs = [circle_pt(i) for i in range(1, m + 1)]
eta = 1 - max(a[0] * b[0] + a[1] * b[1] for a, b in itertools.permutations(xs, 2))
r = 1 - eta / 8
units40 = []
for j in range(kappa):
    for i in range(m):
        w = [Fr(0)] * (2 * kappa)
        w[2 * j], w[2 * j + 1] = xs[i]
        units40.append((tuple(w), -r))
I40 = build({frozenset(((0, a), (1, b))) for a, b in [(1, 2), (2, 1), (2, 3), (3, 2)]}, kappa=kappa, per_color=m)
nets["D40 (kappa 2, m 3)"] = dict(units=units40, readers=[[Fr(1)] * len(units40)], families=[list(range(len(units40)))],
                                  pts=I40["pts"])

out = {}
for name, N in nets.items():
    U, pts = N["units"], N["pts"]
    q2 = []
    for fam in N["families"]:
        for a, b in itertools.combinations(fam, 2):
            wit = witness(U[a], U[b], pts)
            q2.append(dict(pair=[a, b], witness=None if wit is None else [str(c) for c in wit],
                           lp_overlap=lp_overlap(U[a], U[b], pts)))
    q3 = []
    for ri, rw in enumerate(N["readers"]):
        for a, b in itertools.combinations(range(len(U)), 2):
            if rw[a] * rw[b] < 0:
                wit = witness(U[a], U[b], pts)
                q3.append(dict(reader=ri, pair=[a, b], witness=None if wit is None else [str(c) for c in wit],
                               lp_overlap=lp_overlap(U[a], U[b], pts)))
    q2_fail = [x for x in q2 if x["witness"] is not None]
    q3_fail = [x for x in q3 if x["witness"] is not None]
    q2_lp_max = max([x["lp_overlap"] for x in q2], default=0.0)
    q3_lp_max = max([x["lp_overlap"] for x in q3], default=0.0)
    out[name] = dict(Q2="fails" if q2_fail else ("empty" if not q2 else "holds"),
                     Q3="fails" if q3_fail else ("empty" if not q3 else "holds"),
                     Q2_first_witness=q2_fail[0] if q2_fail else None, Q3_first_witness=q3_fail[0] if q3_fail else None,
                     Q2_lp_max_overlap=q2_lp_max, Q3_lp_max_overlap=q3_lp_max, n_units=len(U), n_points=len(pts))
    print(f"{name}: Q2 {out[name]['Q2']} (lp max {q2_lp_max:.3g}), Q3 {out[name]['Q3']} (lp max {q3_lp_max:.3g})"
          + (f"; Q2 witness {q2_fail[0]['pair']} at {q2_fail[0]['witness']}" if q2_fail else "")
          + (f"; Q3 witness reader {q3_fail[0]['reader']} pair {q3_fail[0]['pair']} at {q3_fail[0]['witness']}"
             if q3_fail else ""))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "ta5_principles.json"), "w"), indent=1, default=str)
