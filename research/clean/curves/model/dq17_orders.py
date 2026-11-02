# serves: DQ17
"""Conjecture DQ17: the wall program at k = 1 of Theorem TZ17, on a tree pattern: is a least order always one where each subtree's leaves are consecutive? Exhaustive over orders at K = 4, 5, 6, geometric program per order, solved in log gaps (convex).
Estimator, fixed before the run: 60 random monotone patterns n_ci = exp(f_lca(c, i)), f rising downward with steps up to s = 1, 3 (no scales), plus a control with the same numbers on a shuffled tree against the original tree's compatibility test. Reported: how often the best compatible order equals the best order within 1e-7, and the worst ratio. Then the same with random row and column scales, to see whether scales break it.
Prediction, written before the run: no scales: compatible order is best in every case; control: often not; with scales: fails sometimes."""
import itertools

import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(17)


def value(order, n):
    """TZ17's objective for one order, least over gaps u > 0, n[c, i] for original class labels."""
    K = len(order)
    nn = n[np.ix_(order, order)]

    def F(lu):
        u = np.exp(lu)
        x = np.concatenate([[0], np.cumsum(u)])
        tot = 0.
        for c in range(K):
            for i in range(c + 1, K):
                js = np.arange(c, i)
                tot += nn[c, i] * ((i - c - 1) + 2 * ((x[js] - x[c]) / u[js]).sum())
                tot += nn[i, c] * ((i - c - 1) + 2 * ((x[i] - x[js + 1]) / u[js]).sum())
        return tot
    r = minimize(F, np.zeros(K - 1), method="BFGS", options=dict(gtol=1e-8, maxiter=400))
    return float(r.fun)


def tree(K):
    nodes = [[i] for i in range(K)]
    merges = []
    while len(nodes) > 1:
        i, j = sorted(rng.choice(len(nodes), 2, replace=False))
        b, a = nodes.pop(j), nodes.pop(i)
        merges.append(a + b)
        nodes.append(a + b)
    return merges


def pattern(K, merges, s):
    f = np.zeros((K, K))
    # merges run from deep to the root; depth from the root: later merges get smaller f
    level = {tuple(sorted(m)): None for m in merges}
    val = 0.
    for m in reversed(merges):
        val_m = val + float(rng.uniform(0, s))
        level[tuple(sorted(m))] = val_m
        val = val_m
    seen = {}
    for m in merges:
        fm = level[tuple(sorted(m))]
        for i in m:
            for j in m:
                if i != j and (i, j) not in seen:
                    f[i, j] = fm
                    seen[(i, j)] = True
    return f


def compatible(order, merges):
    pos = {c: i for i, c in enumerate(order)}
    for m in merges:
        ps = sorted(pos[c] for c in m)
        if ps[-1] - ps[0] != len(ps) - 1:
            return False
    return True


def run(K, n, merges):
    best, bestc = np.inf, np.inf
    for order in itertools.permutations(range(K)):
        if order[0] > order[-1]:
            continue
        v = value(list(order), n)
        best = min(best, v)
        if compatible(order, merges):
            bestc = min(bestc, v)
    return best, bestc


out = dict()
for label, scales, shuffle in (("no scales", False, False), ("shuffled tree", False, True), ("scales", True, False)):
    hit, tot, worst = 0, 0, 1.
    for K in (4, 5, 6):
        for _ in range(20 if K < 6 else 6):
            merges = tree(K)
            for s in (1., 3.):
                f = pattern(K, merges, s)
                n = np.exp(f)
                np.fill_diagonal(n, 0)
                if scales:
                    u_, v_ = np.exp(rng.normal(size=K)), np.exp(rng.normal(size=K))
                    n = n * u_[:, None] * v_[None, :]
                test = tree(K) if shuffle else merges
                b, c = run(K, n, test)
                tot += 1
                hit += c <= b * (1 + 1e-7)
                worst = max(worst, c / b)
    out[label] = dict(cases=tot, compatible_is_best=hit, worst_ratio=worst)
    print(label, out[label], flush=True)
