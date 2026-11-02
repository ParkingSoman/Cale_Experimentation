# serves: TM5
# v2 (after referee round 3): two-sided blend claim, and both logit orders (sign of t_S flipped).
"""Exact check of Theorem TM5 (COARSE-THEORY.md B.M): on detector networks with T43's mixes, for a set S of detectors,
   eps_blend(S) = 0  <=>  S has no bad pair  <=>  T59 part 1's linear program (strict exact merge) is feasible,
and when S has no bad pair, Corollary T59.2's unit reproduces t_S at every point of P.
build, colored_graph and detector are copied from tr_tree_search.py (which copies them from t43_clean_hard.py)."""
import itertools
import json
import math
import os
from fractions import Fraction as Fr

import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))


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
    pts, w = [], []
    for v in verts:
        pts.append(x_of(v)); w.append(Fr(1))
    for u, v in bad:
        pts.append(tuple((a + b) / 2 for a, b in zip(x_of(u), x_of(v)))); w.append(Fr(M))
    return dict(verts=verts, bad=bad, pts=pts, w=w)


def colored_graph(H_edges, kappa, m):
    return {frozenset(((j, a), (jj, b))) for j in range(kappa) for jj in range(kappa) if j != jj
            for a in range(1, m + 1) for b in range(1, m + 1) if frozenset((a, b)) in H_edges}


def detector(kappa, m):
    xs = [circle_pt(i) for i in range(1, m + 1)]
    eta = 1 - max(a[0] * b[0] + a[1] * b[1] for a, b in itertools.permutations(xs, 2))
    r = 1 - eta / 8
    W = []
    for j in range(kappa):
        for i in range(m):
            row = [Fr(0)] * (2 * kappa)
            row[2 * j], row[2 * j + 1] = xs[i]
            W.append(row)
    return dict(W=W, r=r, c=2 / (1 - r), eta=eta)


def in_hull(y, Y):
    """is y a convex combination of the rows of Y? (one LP)"""
    if len(Y) == 0:
        return False
    k = len(Y)
    A_eq = np.vstack([Y.T, np.ones((1, k))])
    b_eq = np.r_[y, 1.0]
    res = linprog(np.zeros(k), A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * k, method="highs")
    return res.status == 0


def strict_merge_lp(Zs, t):
    """T59 part 1: exists (m, beta) with m^T z + beta = t where t > 0 and <= 0 where t = 0?"""
    D = Zs.shape[1]
    pos, zer = t > 1e-12, t <= 1e-12
    A = np.c_[Zs, np.ones(len(t))]
    res = linprog(np.zeros(D + 1), A_eq=A[pos], b_eq=t[pos], A_ub=A[zer] if zer.any() else None,
                  b_ub=np.zeros(zer.sum()) if zer.any() else None, bounds=[(None, None)] * (D + 1), method="highs")
    return res.status == 0


def run(kappa, m, H, n_sample=None, seed=0, sign=1.0):
    G = colored_graph(H, kappa, m)
    inst = build(G, kappa, m)
    det = detector(kappa, m)
    X = np.array([[float(a) for a in p] for p in inst["pts"]])
    w = np.array([float(a) for a in inst["w"]])
    p = w / w.sum()
    Wd = np.array([[float(a) for a in row] for row in det["W"]])
    r, c, eta = float(det["r"]), float(det["c"]), float(det["eta"])
    Z = X @ Wd.T - r
    units = inst["verts"]                       # unit index k <-> vertex (block, direction)
    badset = {frozenset(e) for e in inst["bad"]}
    n_units = len(units)
    subsets = [S for k in range(1, n_units + 1) for S in itertools.combinations(range(n_units), k)]
    if n_sample is not None and len(subsets) > n_sample:
        rng = np.random.default_rng(seed)
        subsets = [subsets[i] for i in rng.choice(len(subsets), n_sample, replace=False)]
    rows, exceptions, max_t592_err = [], 0, 0.0
    for S in subsets:
        S = list(S)
        Zs = Z[:, S]
        t = sign * c * np.maximum(Zs, 0).sum(1)
        eps = 0.0
        for i in range(len(X)):
            if in_hull(Zs[i], Zs[t > t[i] + 1e-12]) or in_hull(Zs[i], Zs[t < t[i] - 1e-12]):
                eps += p[i]
        has_bad = any(frozenset((units[a], units[b])) in badset for a, b in itertools.combinations(S, 2))
        lp = strict_merge_lp(Zs, sign * t)
        ok = ((eps == 0) == (not has_bad) == lp)
        if not ((eps == 0) == (not has_bad)) or not ((not has_bad) == lp):
            exceptions += 1
        if not has_bad:
            J = len(S)
            cp = 4 / eta
            l = cp * (Zs.sum(1) + J * r - 1) + 2
            max_t592_err = max(max_t592_err, float(np.abs(np.maximum(l, 0) - sign * t).max()))
        rows.append(dict(S=[f"{units[k][0]}{units[k][1]}" for k in S], eps_blend=eps, bad_pair=has_bad, lp=lp))
    return dict(sign=sign, kappa=kappa, m=m, H=sorted(sorted(e) for e in H), n_sets=len(subsets), exceptions=exceptions,
                n_no_bad=int(sum(not r_["bad_pair"] for r_ in rows)), max_T59_2_error=max_t592_err,
                examples=rows[:6])


cases = []
for sign in (1.0, -1.0):
    cases += [
        run(2, 2, {frozenset((1, 2))}, sign=sign),
        run(2, 3, {frozenset((1, 2)), frozenset((2, 3))}, sign=sign),
        run(3, 3, {frozenset((1, 2)), frozenset((1, 3)), frozenset((2, 3))}, sign=sign),
        run(3, 4, {frozenset((1, 2)), frozenset((2, 3)), frozenset((3, 4))}, n_sample=300, sign=sign),
    ]
for cs in cases:
    print(f"order={'(ch,0)' if cs['sign'] > 0 else '(0,ch)'} kappa={cs['kappa']} m={cs['m']} sets={cs['n_sets']} no-bad-pair sets={cs['n_no_bad']} "
          f"exceptions={cs['exceptions']} max T59.2 error={cs['max_T59_2_error']:.2e}")
ok = all(cs["exceptions"] == 0 and cs["max_T59_2_error"] < 1e-9 for cs in cases)
print("PREDICTION (eps_blend = 0 <=> no bad pair <=> LP feasible; T59.2 exact):", "HELD" if ok else "FAILED")
json.dump(cases, open(os.path.join(HERE, "tm5_blend.json"), "w"), indent=1, default=str)
