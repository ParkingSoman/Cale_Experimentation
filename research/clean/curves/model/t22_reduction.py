# serves: T22
# serves: T22a
"""Check of Lemma T22a and Theorem T22 (COARSE-THEORY.md B.T) on two small graphs.

kappa = 2 colors, two vertices per color (N_V = 4), points in Q^4.
  yes-graph: one edge a1-b2 (a colorful 2-clique exists).
  no-graph:  no edge between the colors (no colorful 2-clique).
Steps:
 1. Build the points, labels, weights and constants of the proof of T22 in exact rationals; check the points are
    distinct.
 2. Build the teacher of Lemma T22a in exact rationals and check Delta_N(x_i) = t_i at every point.
 3. Build B* for the yes-graph and check L(B*) = Gamma (= (N_V - kappa) l_0) with interval arithmetic.
 4. Solve TRUNK-1 by the exact rule of T21 (regions x signs x convex pieces; solver accuracy, not exact) on both
    graphs, for KL and for D2. Read the set C = {v : Delta_B(x_v) > delta} off the least build.
Prediction (written before the run): step 4 gives, on the yes-graph, a least value <= Gamma / Z and C = {a1, b2};
on the no-graph, a least value >= (Gamma + 1/16) / Z and C not a colorful clique.
"""
import itertools
import json
import math
import os
import sys
from fractions import Fraction as Fr

import numpy as np
from mpmath import iv

sys.path.insert(0, os.path.dirname(__file__))
from trunk_exact import exact_min  # noqa: E402

iv.dps = 40


def circle_pt(i):
    return (Fr(1 - i * i, 1 + i * i), Fr(2 * i, 1 + i * i))


def build(edges, kappa=2, per_color=2):
    verts = [(c, i) for c in range(kappa) for i in range(1, per_color + 1)]
    NV = len(verts)

    def x_of(v):
        c, i = v
        x = [Fr(0)] * (2 * kappa)
        x[2 * c], x[2 * c + 1] = circle_pt(i)
        return tuple(x)

    bad = [(u, v) for u, v in itertools.combinations(verts, 2)
           if u[0] == v[0] or (frozenset((u, v)) not in edges)]
    delta = Fr(1, 16 * (NV - kappa + 1))
    M = math.ceil(Fr(12 * (NV + 1)) / delta ** 2)
    pts, lab, w = [], [], []
    for v in verts:
        pts.append(x_of(v)); lab.append(Fr(2)); w.append(Fr(1))
    for u, v in bad:
        pts.append(tuple((a + b) / 2 for a, b in zip(x_of(u), x_of(v)))); lab.append(Fr(0)); w.append(Fr(M))
    assert len(set(pts)) == len(pts), "points not distinct"
    Z = sum(w)
    return dict(verts=verts, NV=NV, kappa=kappa, bad=bad, delta=delta, M=M, pts=pts, lab=lab, w=w, Z=Z, x_of=x_of)


def teacher(pts, lab):
    """Lemma T22a in exact rationals: returns a function x -> Delta_N(x)."""
    n, d = len(pts), len(pts[0])
    for tau in range(1, (d - 1) * n * n + 2):
        v = [Fr(tau) ** l for l in range(d)]
        s = [sum(a * b for a, b in zip(v, p)) for p in pts]
        if len(set(s)) == n:
            break
    order = sorted(range(n), key=lambda i: s[i])
    ss = [s[i] for i in order]
    tt = [lab[i] for i in order]
    mu = [Fr(0)] + [(tt[j + 1] - tt[j]) / (ss[j + 1] - ss[j]) for j in range(n - 1)]
    coef = [mu[j + 1] - mu[j] for j in range(n - 1)]  # unit j uses kink ss[j]

    def gap(x):
        z = sum(a * b for a, b in zip(v, x))
        return tt[0] + sum(c * max(z - ss[j], Fr(0)) for j, c in enumerate(coef))
    return gap, n - 1


def ivsig(t):
    return 1 / (1 + iv.exp(-t))


def ivkl(f, t):
    q, s = ivsig(t), ivsig(f)
    return q * iv.log(q / s) + (1 - q) * iv.log((1 - q) / (1 - s))


out = {}
yes_edges = {frozenset(((0, 1), (1, 2)))}
no_edges = set()
for name, edges in (("yes", yes_edges), ("no", no_edges)):
    I = build(edges)
    gapN, width = teacher(I["pts"], I["lab"])
    teach_ok = all(gapN(p) == t for p, t in zip(I["pts"], I["lab"]))
    rec = {"n_points": len(I["pts"]), "teacher_width": width, "teacher_exact": teach_ok, "M": I["M"],
           "delta": str(I["delta"])}
    print(f"[{name}] {len(I['pts'])} points in Q^4, teacher width {width}, teacher exact on all points: {teach_ok}")
    l0 = ivkl(iv.mpf(0), iv.mpf(2))
    Gamma_kl = (I["NV"] - I["kappa"]) * l0
    if name == "yes":
        # B*: clique {(0,1), (1,2)}: w = (2/eta)(x~_1, x~_2), b = 1 - 2/eta
        cps = [circle_pt(i) for i in (1, 2)]
        eta = 1 - max(a[0] * b[0] + a[1] * b[1] for a, b in itertools.permutations(cps, 2))
        wv = [Fr(2) / eta * c for c in (cps[0] + cps[1])]
        bb = 1 - Fr(2) / eta
        gB = [2 * max(sum(a * b for a, b in zip(wv, p)) + bb, Fr(0)) for p in I["pts"]]
        LB = sum(iv.mpf(wi.numerator) * ivkl(iv.mpf(g.numerator) / g.denominator, iv.mpf(t.numerator))
                 for wi, g, t in zip(I["w"], gB, I["lab"]))
        rec["L(B*) in Gamma"] = bool((LB - Gamma_kl).a <= 1e-30 and (LB - Gamma_kl).b >= -1e-30)
        print(f"[{name}] L(B*) - Gamma encloses 0: {rec['L(B*) in Gamma']}")
    X = np.array([[float(c) for c in p] for p in I["pts"]])
    pi = np.array([float(wi / I["Z"]) for wi in I["w"]])
    t = np.array([float(x) for x in I["lab"]])
    for kind in ("kl", "d2"):
        (v, z, ch), Xt = exact_min(X, pi, t, 1, kind, starts=3)
        D = Xt.shape[1]
        s, A = ch[0]
        gap = s * np.where([i in A for i in range(len(t))], Xt @ z[:D], 0.0) + z[-1]
        C = [I["verts"][i] for i in range(I["NV"]) if gap[i] > float(I["delta"])]
        colorful = (len(C) == I["kappa"] and len({c for c, _ in C}) == I["kappa"]
                    and all(frozenset((u, w_)) in edges for u, w_ in itertools.combinations(C, 2)))
        Gamma = float(Gamma_kl.mid) if kind == "kl" else float(I["NV"] - I["kappa"])
        LZ = v * float(I["Z"])
        rec[kind] = {"L_min": LZ, "Gamma": Gamma, "Gamma+1/16": Gamma + 1 / 16, "C": [str(c) for c in C],
                     "C_is_colorful_clique": colorful}
        print(f"[{name}] {kind}: least L = {LZ:.6f}, Gamma = {Gamma:.6f}, Gamma + 1/16 = {Gamma + 1/16:.6f}, "
              f"C = {C}, colorful clique: {colorful}")
    out[name] = rec

ok = (out["yes"]["teacher_exact"] and out["no"]["teacher_exact"] and out["yes"]["L(B*) in Gamma"]
      and all(out["yes"][k]["C_is_colorful_clique"] and out["yes"][k]["L_min"] <= out["yes"][k]["Gamma"] + 1e-6
              for k in ("kl", "d2"))
      and all((not out["no"][k]["C_is_colorful_clique"]) and out["no"][k]["L_min"] >= out["no"][k]["Gamma+1/16"] - 1e-6
              for k in ("kl", "d2")))
out["all_ok"] = ok
print("ALL OK:", ok)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "t22_reduction.json"), "w"), indent=1, default=str)
