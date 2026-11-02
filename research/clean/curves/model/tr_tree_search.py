# serves: D52
"""Conjecture D52 (COARSE-THEORY.md D.13): does some zoom make step-by-step (class (r)) reach the least value?

Instance: the detector network N^1 of Definition D40 with kappa = 2 blocks and m = 2 directions (4 detectors
A = (0,1), B = (0,2), C = (1,1), D = (1,2)), and the mix of Theorem T43 for the graph H = one edge {1, 2}: the 4 vertex
points (gap 2) and the midpoints of the 4 bad pairs (gap 0, weight M). Good pairs (blend not in the mix): A-D, B-C.
Objective D2 (T43 part 3), weights rescaled so that a vertex point has weight 1 (same order of builds).

1. (a) = least width-1 build over any units (exact pieces, model/trunk_exact.py, Theorem T21).
2. (b) at the top = least unit reading all four pre-activations (should equal (a): Proposition T57).
3. (r) along every zoom with one merge of two current units per step (18 merge orders). Each step solves every piece
   (sign, region of the arrangement of the 8 points in the children's pre-activation space) with the other units
   frozen (free output weights) and branches on every piece that reaches the step's least value. Within a piece the
   solver's point is used, so "not reached" means "not reached by these choices", not a proof.
Numbers are from a convex solver (SLSQP) on each piece, accurate to its tolerance, not exact.
"""
import itertools
import json
import os
import sys

import numpy as np
from scipy.optimize import linprog, minimize

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from trunk_exact import exact_min  # noqa: E402
import math  # noqa: E402
from fractions import Fraction as Fr  # noqa: E402


# build, colored_graph and detector are copied from model/t43_clean_hard.py (that script runs its checks on import)
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
    return dict(verts=verts, NV=NV, kappa=kappa, bad=bad, delta=delta, M=M, pts=pts, lab=lab, w=w)


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
    return dict(W=W, r=r, c=2 / (1 - r))

TOL = 1e-6


def instance():
    kappa, m = 2, 2
    G = colored_graph({frozenset((1, 2))}, kappa, m)
    inst = build(G, kappa, m)
    det = detector(kappa, m)
    X = np.array([[float(a) for a in p] for p in inst["pts"]])
    t = np.array([float(a) for a in inst["lab"]])
    w = np.array([float(a) for a in inst["w"]])
    pi = w / w.min()
    Wd = np.array([[float(a) for a in row] for row in det["W"]])
    r = float(det["r"])
    Z = X @ Wd.T - r                                     # (n, 4) pre-activations of the 4 detectors
    names = ["A", "B", "C", "D"]                          # (block, direction): A=(0,1) B=(0,2) C=(1,1) D=(1,2)
    gapN = float(det["c"]) * np.maximum(Z, 0).sum(1)
    assert np.allclose(gapN, t), "N's gap must equal T43's labels"
    return X, t, pi, Z, names, inst


def regions_free(Zeta):
    n, D = Zeta.shape
    out = {frozenset()}
    for signs in itertools.product([1, -1], repeat=n):
        sg = np.array(signs, float)
        res = linprog(np.zeros(D), A_ub=-(sg[:, None] * Zeta), b_ub=-np.ones(n), bounds=[(None, None)] * D,
                      method="highs")
        if res.status == 0:
            out.add(frozenset(np.where(sg > 0)[0]))
    return sorted(out, key=lambda a: (len(a), sorted(a)))


def solve_piece(Zeta, F, pi, t, s, A, starts=3, rng=None):
    """min over u of 1/4 sum pi (gap - t)^2, gap = s mask_A (Zeta u) + F beta + gamma, beta, gamma free."""
    n, D = Zeta.shape
    mask = np.array([1.0 if i in A else 0.0 for i in range(n)])
    Q = np.hstack([F, np.ones((n, 1))]) if F.size else np.ones((n, 1))
    Pi = np.diag(pi)
    Pw = np.eye(n) - Q @ np.linalg.pinv(Q.T @ Pi @ Q) @ Q.T @ Pi
    Mm = Pw @ (s * mask[:, None] * Zeta)
    tt = Pw @ t

    def f(u):
        rr = Mm @ u - tt
        return 0.25 * float(rr @ (pi * rr))

    def g(u):
        rr = Mm @ u - tt
        return 0.5 * Mm.T @ (pi * rr)
    C = np.array([Zeta[i] if i in A else -Zeta[i] for i in range(n)])
    con = {"type": "ineq", "fun": lambda u: C @ u, "jac": lambda u: C}
    rng = rng or np.random.default_rng(0)
    best = (np.inf, None)
    for k in range(starts):
        u0 = np.zeros(D) if k == 0 else rng.normal(size=D)
        res = minimize(f, u0, jac=g, constraints=[con], method="SLSQP", options={"maxiter": 3000, "ftol": 1e-15})
        if np.all(C @ res.x >= -1e-8) and res.fun < best[0]:
            best = (res.fun, res.x)
    return best


_reg_cache = {}


def step(Zeta, F, pi, t):
    key = Zeta.round(12).tobytes()
    if key not in _reg_cache:
        _reg_cache[key] = regions_free(Zeta)
    sols = []
    for A in _reg_cache[key]:
        for s in (1, -1):
            v, u = solve_piece(Zeta, F, pi, t, s, A)
            if u is not None:
                sols.append((v, s, A, u))
    vmin = min(v for v, *_ in sols)
    tied = [x for x in sols if x[0] <= vmin + TOL * max(1.0, abs(vmin))]
    # keep one solution per distinct unit output on the points (the data vector relu(Zeta u), up to scale)
    seen, out = set(), []
    for v, s, A, u in tied:
        h = np.maximum(Zeta @ u, 0)
        hn = h / h.max() if h.max() > 1e-9 else h * 0
        k = tuple(np.round(hn, 5))
        if k not in seen:
            seen.add(k)
            out.append((v, s, A, u))
    return vmin, out


def search(nodes, pi, t, path, results, depth_cap=200000):
    """nodes: list of (label, p, h). One merge per step, branch on tied least pieces."""
    if len(nodes) == 1:
        results.append((path, path[-1][1]))
        return
    for a, b in itertools.combinations(range(len(nodes)), 2):
        la, pa, _ = nodes[a]
        lb, pb, _ = nodes[b]
        Zeta = np.column_stack([pa, pb, np.ones(len(t))])
        rest = [nodes[i] for i in range(len(nodes)) if i not in (a, b)]
        F = np.column_stack([h for _, _, h in rest]) if rest else np.zeros((len(t), 0))
        vmin, tied = step(Zeta, F, pi, t)
        for v, s, A, u in tied:
            p = Zeta @ u
            lab = f"({la}{lb})"
            newnodes = rest + [(lab, p, np.maximum(p, 0))]
            grouping = frozenset(frozenset(c for c in L if c.isalpha()) for L, _, _ in newnodes if len(L) > 1)
            search(newnodes, pi, t, path + [(lab, v, sorted(A), grouping)], results)
            if len(results) > depth_cap:
                return


def vb_joint(groups, Z, names, pi, t, starts=2):
    """V_b at a grouping: one new unit per group (reads that group's pre-activations), other units kept. D2."""
    n = len(t)
    idx = {c: i for i, c in enumerate(names)}
    kept = [i for i, c in enumerate(names) if not any(c in g for g in groups)]
    F = np.column_stack([np.maximum(Z[:, i], 0) for i in kept]) if kept else np.zeros((n, 0))
    Zetas = [np.column_stack([Z[:, idx[c]] for c in sorted(g)] + [np.ones(n)]) for g in groups]
    for Ze in Zetas:
        key = Ze.round(12).tobytes()
        if key not in _reg_cache:
            _reg_cache[key] = regions_free(Ze)
    choices = [[(sg, A) for A in _reg_cache[Ze.round(12).tobytes()] for sg in (1, -1)] for Ze in Zetas]
    Q = np.hstack([F, np.ones((n, 1))])
    Pw = np.eye(n) - Q @ np.linalg.pinv(Q.T @ np.diag(pi) @ Q) @ Q.T @ np.diag(pi)
    tt = Pw @ t
    dims = [Ze.shape[1] for Ze in Zetas]
    offs = np.concatenate([[0], np.cumsum(dims)]).astype(int)
    best = np.inf
    rng = np.random.default_rng(1)
    for combo in itertools.product(*choices):
        Mm = np.hstack([Pw @ (sg * np.array([1.0 if i in A else 0.0 for i in range(n)])[:, None] * Ze)
                        for (sg, A), Ze in zip(combo, Zetas)])
        rows = []
        for g, ((sg, A), Ze) in enumerate(zip(combo, Zetas)):
            for i in range(n):
                row = np.zeros(offs[-1]); row[offs[g]:offs[g + 1]] = Ze[i] if i in A else -Ze[i]; rows.append(row)
        C = np.array(rows)
        f = lambda u: 0.25 * float((Mm @ u - tt) @ (pi * (Mm @ u - tt)))
        gr = lambda u: 0.5 * Mm.T @ (pi * (Mm @ u - tt))
        con = {"type": "ineq", "fun": lambda u: C @ u, "jac": lambda u: C}
        for k in range(starts):
            u0 = np.zeros(offs[-1]) if k == 0 else rng.normal(size=offs[-1])
            r = minimize(f, u0, jac=gr, constraints=[con], method="SLSQP", options={"maxiter": 3000, "ftol": 1e-15})
            if np.all(C @ r.x >= -1e-8) and r.fun < best:
                best = r.fun
    return best


def main():
    X, t, pi, Z, names, inst = instance()
    (va, _, cha), _ = exact_min(X, pi, t, 1, kind="d2", starts=4)
    Zeta_all = np.hstack([Z, np.ones((len(t), 1))])
    vb, tied_b = step(Zeta_all, np.zeros((len(t), 0)), pi, t)
    leaves = [(names[i], Z[:, i], np.maximum(Z[:, i], 0)) for i in range(4)]
    results = []
    search(leaves, pi, t, [], results)
    # V_b at every grouping met along the zooms (the top grouping is the one group of all four: vb above)
    VB = {}
    for path, _ in results:
        for _, _, _, grp in path:
            if grp not in VB:
                VB[grp] = vb if len(grp) == 1 and len(next(iter(grp))) == 4 else vb_joint(sorted(map(tuple, grp)), Z, names, pi, t)
    # per branch: the largest gap over levels; per zoom: the least such gap over branches (eps_recur of Part C)
    by_tree = {}
    for path, v in results:
        key = " -> ".join(lab for lab, _, _, _ in path)
        gap = max(lv - VB[grp] for _, lv, _, grp in path)
        by_tree.setdefault(key, []).append((v, gap))
    rows = []
    for key, vals in sorted(by_tree.items()):
        tops = [a for a, _ in vals]; gaps = [b for _, b in vals]
        rows.append(dict(zoom=key, branches=len(vals), best_top=min(tops), worst_top=max(tops),
                         reaches_least=bool(min(tops) <= va + TOL * max(1, va)),
                         always_reaches=bool(max(tops) <= va + TOL * max(1, va)),
                         eps_levels_best_branch=min(gaps), every_level_optimal=bool(min(gaps) <= 1e-4 * max(1, va))))
    first = {}
    for path, v in results:
        first.setdefault(path[0][0], path[0][1])
    out = dict(
        instance="D40 kappa=2 m=2, H = one edge; good pairs A-D, B-C; objective D2 (vertex weight 1, midpoint weight M)",
        M=int(inst["M"]), least_any_unit_a=va, least_b_top=vb, b_equals_a=bool(abs(vb - va) <= TOL * max(1, va)),
        first_merge_cost=first, zooms=rows,
        n_zooms_some_branch_reaches=sum(r["reaches_least"] for r in rows),
        n_zooms_every_branch_reaches=sum(r["always_reaches"] for r in rows), n_zooms=len(rows),
        vb_by_grouping={" ".join("".join(sorted(g)) for g in sorted(map(sorted, k))): v for k, v in VB.items()},
        n_zooms_every_level_optimal=sum(r["every_level_optimal"] for r in rows),
        eps_recur=min(r["eps_levels_best_branch"] for r in rows))
    with open(os.path.join(HERE, "tr_tree_search.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"(a) least width-1 = {va:.6f}   (b) top = {vb:.6f}   b = a: {out['b_equals_a']}")
    print("first-merge step cost:", {k: round(v, 6) for k, v in first.items()})
    for r_ in rows:
        print(f"{r_['zoom']:<28} branches {r_['branches']:>2}  top best {r_['best_top']:.6f} worst {r_['worst_top']:.6f}"
              f"  reaches {r_['reaches_least']}  always {r_['always_reaches']}  level-gap {r_['eps_levels_best_branch']:.6f}")
    print("V_b by grouping:", {k: round(v, 6) for k, v in out["vb_by_grouping"].items()})
    print("zooms with a branch optimal at EVERY level:", out["n_zooms_every_level_optimal"], "eps_recur =", out["eps_recur"])
    print("zooms where some branch reaches the least:", out["n_zooms_some_branch_reaches"], "of", out["n_zooms"],
          "| every branch:", out["n_zooms_every_branch_reaches"])


if __name__ == "__main__":
    main()
