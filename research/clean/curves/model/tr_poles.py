# serves: D54
"""Definition D54 (COARSE-THEORY.md B.W.2): what kind of unit do least builds use? One exploration run.

1. Detector network of T61 (kappa = 2, m = 2, H one edge): every grouping whose groups have target size 1 (the three
   pairings, the four "three units + one kept", the whole layer). For each: every zero-loss piece (a linear program
   per piece; a feasible point is rounded to fractions and checked exactly), or, if none, a least build (numeric).
   Each new unit is classified: presence / split / dial-or-detector (D54, D31).
2. T20 (d = 1, OR pair) and T33 (four units in R^3, width 1): their known least units, classified the same way.
"""
import itertools
import json
import os
import sys
from fractions import Fraction as Fr

import numpy as np
from scipy.optimize import linprog, minimize

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import tr_tree_search as T  # noqa: E402


def classify(pre, Zmem, w):
    """pre: unit pre-activation on P; Zmem: member pre-activations (n x s); w: point weights."""
    fires = pre > 1e-9
    Pi = [Zmem[:, i] > 1e-9 for i in range(Zmem.shape[1])]
    union = np.any(np.column_stack(Pi), axis=1) if Pi else np.zeros_like(fires)
    presence = bool(np.all(~fires | union))
    split = any(np.any(fires & Pi[i]) and np.all(pre[Pj] <= 1e-9) and np.any(Pj)
                for i in range(len(Pi)) for j, Pj in enumerate(Pi) if j != i)
    share = float(w[fires].sum() / w.sum())
    return dict(presence=presence, split=bool(split), dial=share >= 0.5, firing_share=round(share, 4),
                fires_on=[int(k) for k in np.where(fires)[0]])


def zero_loss_pieces(groups, Z, names, t):
    n = len(t)
    idx = {c: i for i, c in enumerate(names)}
    kept = [i for i, c in enumerate(names) if not any(c in g for g in groups)]
    Zetas = [np.column_stack([Z[:, idx[c]] for c in g] + [np.ones(n)]) for g in groups]
    regs = [T.regions_free(Ze) for Ze in Zetas]
    out = []
    for combo in itertools.product(*[[(s, A) for A in R for s in (1, -1)] for R in regs]):
        cols, rows = [], []
        dims = [Ze.shape[1] for Ze in Zetas]
        offs = np.concatenate([[0], np.cumsum(dims)]).astype(int)
        D = offs[-1] + len(kept) + 1
        for g, ((s, A), Ze) in enumerate(zip(combo, Zetas)):
            mask = np.array([1.0 if i in A else 0.0 for i in range(n)])
            blk = np.zeros((n, D)); blk[:, offs[g]:offs[g + 1]] = s * mask[:, None] * Ze; cols.append(blk)
            for i in range(n):
                row = np.zeros(D); row[offs[g]:offs[g + 1]] = -(Ze[i] if i in A else -Ze[i]); rows.append(row)
        Aeq = sum(cols)
        for k, i in enumerate(kept):
            Aeq[:, offs[-1] + k] = np.maximum(Z[:, i], 0)
        Aeq[:, -1] = 1
        r = linprog(np.zeros(D), A_ub=np.array(rows), b_ub=np.zeros(len(rows)), A_eq=Aeq, b_eq=t,
                    bounds=[(None, None)] * D, method="highs")
        if r.status == 0:
            out.append((combo, r.x, offs, Zetas))
    return out


def least_build(groups, Z, names, pi, t):
    """Numeric least (b) build for D2 at the grouping, with its units' pre-activations."""
    n = len(t)
    idx = {c: i for i, c in enumerate(names)}
    kept = [i for i, c in enumerate(names) if not any(c in g for g in groups)]
    F = np.column_stack([np.maximum(Z[:, i], 0) for i in kept]) if kept else np.zeros((n, 0))
    Zetas = [np.column_stack([Z[:, idx[c]] for c in g] + [np.ones(n)]) for g in groups]
    regs = [T.regions_free(Ze) for Ze in Zetas]
    Q = np.hstack([F, np.ones((n, 1))])
    Pw = np.eye(n) - Q @ np.linalg.pinv(Q.T @ np.diag(pi) @ Q) @ Q.T @ np.diag(pi)
    tt = Pw @ t
    dims = [Ze.shape[1] for Ze in Zetas]
    offs = np.concatenate([[0], np.cumsum(dims)]).astype(int)
    best = (np.inf, None, None)
    for combo in itertools.product(*[[(s, A) for A in R for s in (1, -1)] for R in regs]):
        Mm = np.hstack([Pw @ (s * np.array([1.0 if i in A else 0.0 for i in range(n)])[:, None] * Ze)
                        for (s, A), Ze in zip(combo, Zetas)])
        C = np.array([np.r_[np.zeros(offs[g]), (Ze[i] if i in A else -Ze[i]), np.zeros(offs[-1] - offs[g + 1])]
                      for g, ((s, A), Ze) in enumerate(zip(combo, Zetas)) for i in range(n)])
        f = lambda u: 0.25 * float((Mm @ u - tt) @ (pi * (Mm @ u - tt)))
        g_ = lambda u: 0.5 * Mm.T @ (pi * (Mm @ u - tt))
        r = minimize(f, np.zeros(offs[-1]), jac=g_, constraints=[{"type": "ineq", "fun": lambda u: C @ u, "jac": lambda u: C}],
                     method="SLSQP", options={"maxiter": 3000, "ftol": 1e-15})
        if np.all(C @ r.x >= -1e-8) and r.fun < best[0]:
            best = (r.fun, r.x, combo)
    v, u, combo = best
    pres = [Zetas[g] @ u[offs[g]:offs[g + 1]] for g in range(len(groups))]
    return v, pres, combo


def detector_part():
    X, t, pi, Z, names, inst = T.instance()
    w = pi
    groupings = [[("A", "C"), ("B", "D")], [("A", "B"), ("C", "D")], [("A", "D"), ("B", "C")]]
    groupings += [[g] for g in itertools.combinations("ABCD", 3)] + [[("A", "B", "C", "D")]]
    idx = {c: i for i, c in enumerate(names)}
    rows = []
    for G in groupings:
        lab = " ".join("".join(g) for g in G)
        pieces = zero_loss_pieces(G, Z, names, t)
        rec = dict(grouping=lab, zero_loss_pieces=len(pieces))
        if pieces:
            kinds = []
            for combo, x, offs, Zetas in pieces:
                units = []
                for g, grp in enumerate(G):
                    pre = Zetas[g] @ x[offs[g]:offs[g + 1]]
                    units.append(dict(group="".join(grp), sign=combo[g][0],
                                      **classify(pre, np.column_stack([Z[:, idx[c]] for c in grp]), w)))
                kinds.append(units)
            rec["any_all_presence"] = any(all(u["presence"] for u in us) for us in kinds)
            rec["every_build_has_split"] = all(any(u["split"] for u in us) for us in kinds)
            rec["pieces"] = kinds
        else:
            v, pres, combo = least_build(G, Z, names, pi, t)
            rec["least_value_vertex_units"] = v
            rec["least_build_units"] = [dict(group="".join(grp), sign=combo[g][0],
                                             **classify(pres[g], np.column_stack([Z[:, idx[c]] for c in grp]), w))
                                        for g, grp in enumerate(G)]
        rows.append(rec)
        print(lab, "| zero-loss pieces:", rec["zero_loss_pieces"],
              "| some all-presence build:" if pieces else "| least value:",
              rec.get("any_all_presence", round(rec.get("least_value_vertex_units", -1), 6)),
              "| every build has a split unit:" if pieces else "", rec.get("every_build_has_split", ""))
    return rows


def t20_part():
    xs = np.arange(-3, 4, dtype=float)
    Zm = np.column_stack([xs - 1, -xs - 1])
    pre = xs - 8 / 5
    return dict(instance="T20, least D2 width-1 unit relu(x - 8/5)", **classify(pre, Zm, np.ones(7)))


def t33_part():
    W = np.array([(1, 0, .5), (0, 1, .5), (-1, 0, .5), (0, -1, .5)])
    rot = lambda v: np.array([-v[1], v[0], v[2]])
    seeds = [((-1, 2, -2), 3), ((2, 1, -1), 1), ((1, 0, 2), 2)]
    pts, wts = [], []
    for v, wt in seeds:
        v = np.array(v, float)
        for _ in range(4):
            pts.append(v.copy()); wts.append(wt); v = rot(v)
    P = np.array(pts); wts = np.array(wts, float)
    Zm = P @ W.T - 1
    pre = -P[:, 2] / 2
    return dict(instance="T33, least width-1 unit with gap 1 - relu(-x_3 / 2)", **classify(pre, Zm, wts))


def main():
    rows = detector_part()
    t20 = t20_part(); t33 = t33_part()
    print("T20:", {k: t20[k] for k in ("presence", "split", "dial", "firing_share")})
    print("T33:", {k: t33[k] for k in ("presence", "split", "dial", "firing_share")})
    with open(os.path.join(HERE, "tr_poles.json"), "w") as f:
        json.dump(dict(detector=rows, t20=t20, t33=t33), f, indent=1, default=str)


if __name__ == "__main__":
    main()
