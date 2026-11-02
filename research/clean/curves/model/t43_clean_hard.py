# serves: T43
# serves: T43a
"""Check of Lemma T43a and Theorem T43 (COARSE-THEORY.md B.K) on small instances, kappa = 2.

Instances: Clique with kappa = 2 on H with m = 2 vertices (yes: one edge; no: no edge) and m = 3 vertices
(yes: path 1-2-3; no: no edge). G = the colored graph of the proof (copies of H's vertices in each color).
Steps:
 1. Build T22's points, labels and weights for G in exact rationals (same construction as model/t22_reduction.py).
 2. Build the detector networks N^1 and N^2 of Definition D40 in exact rationals. Check, exactly: the logit difference
    equals T22's label at every point (Lemma T43a part 6); N^2 and N^1 agree at every point (part 5); every point lies in
    the domain X (block norms add to at most 1).
 3. Check exactly that the block swap maps the layer-1 weights to a permutation of themselves (part 1) and maps the mix
    (points with weights) onto itself.
 4. Sample 200000 random points of X and check that no two layer-1 units are on together (part 2; a sanity check, the
    proof is by hand and in Lean).
 5. Solve the width-1 problem with the exact rule of T21 (`trunk_exact.exact_min`; solver accuracy) for KL and D2, read
    the set C = {v : gap > delta}, and compare with the answer for H.
Step 5 runs only for m = 2 (first run stopped: m = 3 has 17 to 21 points, too many for the 2^n enumeration).
Prediction (written before the run): steps 2 to 4 pass with no exception; in step 5, on yes-instances the least value
is at most Gamma / Z (up to solver accuracy) and C is a colorful clique; on no-instances the least value is at least
(Gamma + 1/16) / Z and C is not a colorful clique.
"""
import itertools
import json
import math
import os
import sys
from fractions import Fraction as Fr

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from trunk_exact import exact_min  # noqa: E402


def circle_pt(i):
    return (Fr(1 - i * i, 1 + i * i), Fr(2 * i, 1 + i * i))


def build(edges, kappa, per_color):
    verts = [(c, i) for c in range(kappa) for i in range(1, per_color + 1)]
    NV = len(verts)

    def x_of(v):
        c, i = v
        x = [Fr(0)] * (2 * kappa)
        x[2 * c], x[2 * c + 1] = circle_pt(i)
        return tuple(x)

    bad = [(u, v) for u, v in itertools.combinations(verts, 2) if u[0] == v[0] or (frozenset((u, v)) not in edges)]
    delta = Fr(1, 16 * (NV - kappa + 1))
    M = math.ceil(Fr(12 * (NV + 1)) / delta ** 2)
    pts, lab, w = [], [], []
    for v in verts:
        pts.append(x_of(v)); lab.append(Fr(2)); w.append(Fr(1))
    for u, v in bad:
        pts.append(tuple((a + b) / 2 for a, b in zip(x_of(u), x_of(v)))); lab.append(Fr(0)); w.append(Fr(M))
    assert len(set(pts)) == len(pts)
    return dict(verts=verts, NV=NV, kappa=kappa, bad=bad, delta=delta, M=M, pts=pts, lab=lab, w=w, Z=sum(w))


def colored_graph(H_edges, kappa, m):
    return {frozenset(((j, a), (jj, b))) for j in range(kappa) for jj in range(kappa) if j != jj
            for a in range(1, m + 1) for b in range(1, m + 1) if frozenset((a, b)) in H_edges}


def detector(kappa, m):
    xs = [circle_pt(i) for i in range(1, m + 1)]
    eta = 1 - max(a[0] * b[0] + a[1] * b[1] for a, b in itertools.permutations(xs, 2))
    r = 1 - eta / 8
    W, labels = [], []
    for j in range(kappa):
        for i in range(m):
            row = [Fr(0)] * (2 * kappa)
            row[2 * j], row[2 * j + 1] = xs[i]
            W.append(row); labels.append((j, i))
    c = 2 / (1 - r)

    def layer1(x):
        return [max(sum(a * b for a, b in zip(row, x)) - r, Fr(0)) for row in W]

    def gap(x, L):
        h = layer1(x)
        if L == 1:
            return c * sum(h)
        f = max(sum(h), Fr(0))
        for _ in range(3, L + 1):
            f = max(f, Fr(0))
        return c * f
    return dict(W=W, r=r, eta=eta, c=c, labels=labels, gap=gap, xs=xs)


def is_sq(q):
    n, d = q.numerator, q.denominator
    rn, rd = math.isqrt(n), math.isqrt(d)
    return rn * rn == n and rd * rd == d, Fr(rn, rd)


def in_X(x, kappa):
    """sum of block norms <= 1, exactly (block norms here are rational or the sum has at most one irrational term)."""
    norms2 = [x[2 * j] ** 2 + x[2 * j + 1] ** 2 for j in range(kappa)]
    nz = [q for q in norms2 if q != 0]
    rat = [is_sq(q) for q in nz]
    if all(ok for ok, _ in rat):
        return sum(v for _, v in rat) <= 1
    if len(nz) == 1:
        return nz[0] <= 1
    raise ValueError("unexpected point shape")


def swap(x, kappa, perm):
    y = [Fr(0)] * (2 * kappa)
    for j in range(kappa):
        y[2 * perm[j]], y[2 * perm[j] + 1] = x[2 * j], x[2 * j + 1]
    return tuple(y)


rng = np.random.default_rng(0)
out = {}
cases = [("m2-yes", 2, {frozenset((1, 2))}, True), ("m2-no", 2, set(), False),
         ("m3-yes", 3, {frozenset((1, 2)), frozenset((2, 3))}, True), ("m3-no", 3, set(), False)]
kappa = 2
all_ok = True
for name, m, H, answer in cases:
    I = build(colored_graph(H, kappa, m), kappa, m)
    D = detector(kappa, m)
    rec = {"n_points": len(I["pts"]), "r": str(D["r"]), "eta": str(D["eta"])}
    rec["labels_exact_L1"] = all(D["gap"](p, 1) == t for p, t in zip(I["pts"], I["lab"]))
    rec["labels_exact_L2_L4"] = all(D["gap"](p, L) == D["gap"](p, 1) for p in I["pts"] for L in (2, 4))
    rec["points_in_X"] = all(in_X(p, kappa) for p in I["pts"])
    perm = [1, 0]
    rows = {tuple(r_) for r_ in D["W"]}
    rec["weights_equivariant"] = all(tuple(swap(tuple(r_), kappa, perm)) in rows for r_ in D["W"])
    mix = {(p, w) for p, w in zip(I["pts"], I["w"])}
    rec["mix_invariant"] = all((swap(p, kappa, perm), w) in mix for p, w in mix)
    # step 4: random points of X (uniform direction per block, block norms from a random point of the simplex)
    Wf = np.array([[float(c) for c in r_] for r_ in D["W"]]); rf = float(D["r"])
    k = 200000
    s = rng.dirichlet(np.ones(kappa + 1), size=k)[:, :kappa]  # block norms, sum <= 1
    s = np.maximum(s, rng.uniform(0, 1, size=(k, kappa)) * (s.sum(1, keepdims=True) <= 1))  # mix in larger norms
    s = s / np.maximum(1, s.sum(1, keepdims=True))
    ang = rng.uniform(0, 2 * np.pi, size=(k, kappa))
    Xs = np.zeros((k, 2 * kappa))
    for j in range(kappa):
        Xs[:, 2 * j], Xs[:, 2 * j + 1] = s[:, j] * np.cos(ang[:, j]), s[:, j] * np.sin(ang[:, j])
    on = (Xs @ Wf.T - rf) > 0
    rec["max_units_on_together_sampled"] = int(on.sum(1).max())
    # step 5: exact rule of T21, width 1 (only for m = 2: the rule enumerates 2^n sign patterns, n <= 12)
    if m > 2:
        rec['exact_checks_ok'] = (rec['labels_exact_L1'] and rec['labels_exact_L2_L4'] and rec['points_in_X'] and rec['weights_equivariant'] and rec['mix_invariant'] and rec['max_units_on_together_sampled'] <= 1)
        print(f"[{name}] {len(I['pts'])} points; exact checks only: {rec['exact_checks_ok']}; max on together: {rec['max_units_on_together_sampled']}")
        all_ok = all_ok and rec['exact_checks_ok']; out[name] = rec
        continue
    X = np.array([[float(c) for c in p] for p in I["pts"]])
    pi = np.array([float(wi / I["Z"]) for wi in I["w"]])
    t = np.array([float(x) for x in I["lab"]])
    for kind in ("kl", "d2"):
        (v, z, ch), Xt = exact_min(X, pi, t, 1, kind, starts=3)
        Dd = Xt.shape[1]
        sg, A = ch[0]
        g = sg * np.where([i in A for i in range(len(t))], Xt @ z[:Dd], 0.0) + z[-1]
        C = [I["verts"][i] for i in range(I["NV"]) if g[i] > float(I["delta"])]
        Gedges = colored_graph(H, kappa, m)
        colorful = (len(C) == kappa and len({c_ for c_, _ in C}) == kappa
                    and all(frozenset((a, b)) in Gedges for a, b in itertools.combinations(C, 2)))
        l0 = (math.log1p(math.exp(0)) - (1 / (1 + math.exp(-2))) * 0
              - (-(1 / (1 + math.exp(-2))) * math.log(1 / (1 + math.exp(-2)))
                 - (1 - 1 / (1 + math.exp(-2))) * math.log(1 - 1 / (1 + math.exp(-2)))))
        Gamma = (I["NV"] - kappa) * (l0 if kind == "kl" else 1.0)
        LZ = v * float(I["Z"])
        ok = (colorful and LZ <= Gamma + 1e-5) if answer else ((not colorful) and LZ >= Gamma + 1 / 16 - 1e-5)
        rec[kind] = dict(L_min=LZ, Gamma=Gamma, C=[str(c_) for c_ in C], colorful_clique=colorful, as_predicted=ok)
        print(f"[{name}] {kind}: least L = {LZ:.6f}, Gamma = {Gamma:.6f}, C = {C}, colorful: {colorful}, "
              f"H has a 2-clique: {answer}, as predicted: {ok}")
    exact_ok = (rec["labels_exact_L1"] and rec["labels_exact_L2_L4"] and rec["points_in_X"]
                and rec["weights_equivariant"] and rec["mix_invariant"] and rec["max_units_on_together_sampled"] <= 1)
    rec["exact_checks_ok"] = exact_ok
    print(f"[{name}] {len(I['pts'])} points; r = {D['r']}; exact checks (labels, depth, domain, symmetry): {exact_ok}; "
          f"max units on together in {k} samples: {rec['max_units_on_together_sampled']}")
    all_ok = all_ok and exact_ok and rec["kl"]["as_predicted"] and rec["d2"]["as_predicted"]
    out[name] = rec
out["all_ok"] = all_ok
print("ALL OK:", all_ok)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "t43_clean_hard.json"), "w"), indent=1, default=str)
