# serves: TQ24
"""Proposition TQ24: on random monotone tree patterns, the largest biclique of H_theta (edges where W < theta) by brute force against the formula: blocks to one side, free leaves to both, the best sub-sum.
Estimator, fixed before the run: 400 random binary trees on 4 to 9 leaves, f increasing downward, W_cc at least f at the parent or, in a quarter of the runs on a leaf with no block, below theta; every level theta. Also the referee's two examples: three cherries, largest biclique 8; a product pattern (1, 2, 4, 8) with an induced path at theta = 8.
Prediction, written before the run: 0 mismatches."""
import itertools

import numpy as np

rng = np.random.default_rng(24)


def random_tree(n):
    nodes = [dict(leaves=[i], f=None, kids=[]) for i in range(n)]
    while len(nodes) > 1:
        i, j = sorted(rng.choice(len(nodes), 2, replace=False))
        b, a = nodes.pop(j), nodes.pop(i)
        nodes.append(dict(leaves=a["leaves"] + b["leaves"], f=None, kids=[a, b]))
    return nodes[0]


def assign(node, floor):
    node["f"] = floor + float(rng.uniform(0, 2))
    for k in node["kids"]:
        assign(k, node["f"])


def internal(node):
    if node["kids"]:
        yield node
        for k in node["kids"]:
            yield from internal(k)


def lca_w(root, n):
    W = np.zeros((n, n))

    def rec(node):
        if not node["kids"]:
            return
        a, b = node["kids"]
        for i in a["leaves"]:
            for j in b["leaves"]:
                W[i, j] = W[j, i] = node["f"]
        rec(a)
        rec(b)
    rec(root)
    return W


def parent_f(root, n):
    out = np.zeros(n)

    def rec(node):
        for k in node["kids"]:
            if not k["kids"]:
                out[k["leaves"][0]] = node["f"]
            rec(k)
    rec(root)
    return out


def brute(H):
    n, c = H.shape
    best = 0
    for bits in range(1, 2 ** n):
        rows = [i for i in range(n) if bits >> i & 1]
        cols = int(np.all(H[rows], axis=0).sum())
        best = max(best, len(rows) * cols)
    return best


def formula(root, W, theta, n):
    blocks, seen = [], set()

    def rec(node):
        if not node["kids"]:
            return
        if node["f"] >= theta:
            blocks.append(len(node["leaves"]))
            seen.update(node["leaves"])
            return
        for k in node["kids"]:
            rec(k)
    rec(root)
    free = 0
    for c in range(n):
        if c not in seen:
            if W[c, c] >= theta:
                blocks.append(1)
            else:
                free += 1
    subs = {0}
    for b in blocks:
        subs |= {x + b for x in subs}
    tot = sum(blocks)
    return max((x + free) * (tot - x + free) for x in subs)


fails = 0
runs = 0
for _ in range(400):
    n = int(rng.integers(4, 10))
    root = random_tree(n)
    assign(root, 0.)
    W = lca_w(root, n)
    pf = parent_f(root, n)
    fs = sorted(x["f"] for x in internal(root))
    for theta in fs + [fs[-1] + 1]:
        for c in range(n):
            # a leaf inside a block must have W_cc at least f at its parent; a leaf with no block may be either
            W[c, c] = pf[c] + rng.uniform(0, 1) if (pf[c] >= theta or rng.random() < .75) else theta - 1
        runs += 1
        H = W < theta
        b, g = brute(H), formula(root, W, theta, n)
        if b != g:
            fails += 1
            print("mismatch", n, theta, b, g)
print("runs", runs, "fails", fails)
W = np.ones((6, 6))
for a, b in ((0, 1), (2, 3), (4, 5)):
    W[np.ix_([a, b], [a, b])] = 100.
b3 = brute(W < 100)
print("three cherries, largest biclique of the low-weight graph:", b3)
u = np.array([1., 2., 4., 8.])
G8 = np.outer(u, u) >= 8
print("product pattern, rows 1, 2, columns 3, 4:", G8[0, 3], G8[1, 3], G8[1, 2], G8[0, 2], "(True True True False is an induced path)")
fails += b3 != 8
print("fails", fails)
