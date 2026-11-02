# serves: TY2
"""Exact check of Theorem TY2 and Proposition TY3 (COARSE-THEORY.md B.Y): the merge rule on random synthetic last
layers with positions.

Instance: P = n points with random weights p_x, T positions, C channels, n_y logits. Each channel's pre-activation is
a positive multiple of one of F base functions (so the families are exactly proportional), optionally plus noise of
size sigma; random logit weights W and bias c. Checks, per instance:
  A  exact case (sigma = 0, partition = the families): D_P(B^m) = 0 to rounding
  B  bound (sigma > 0, partition = families, and a coarser partition): D_P(B^m) <= E^2 / 4
  C  rescaling (Fact F1) of one channel: E and y_B^m unchanged
  D  copy (Definition DN1) of one channel into 3 with shares lam: E and y_B^m unchanged
  E  re-fit: D_P(B^m*) <= D_P(B^m) (gradient descent on the convex re-fit problem, started at B^m)
  F  dynamic program of TY3 part 3 = brute force over all cuts of a random binary tree, for every k
Usage: python ty_merge.py [n_instances]
"""
import itertools, sys
import numpy as np

rng = np.random.default_rng(7)
N_INST = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def logsm(y):
    m = y.max(-1, keepdims=True); return y - m - np.log(np.exp(y - m).sum(-1, keepdims=True))


def kl(p, yN, yB):
    a, b = logsm(yN), logsm(yB)
    return float((p * (np.exp(a) * (a - b)).sum(-1)).sum())


def Pperp(v):
    return v - v.mean(0, keepdims=True)


def stats(z, W, p):
    """s_u, c_u, zhat_u, read mask (DM10). z: C x n x T."""
    s = np.sqrt((p[None, :, None] * z ** 2).mean(2).sum(1))
    c = s * np.linalg.norm(Pperp(W), axis=0)
    pooled = np.maximum(z, 0).mean(2)
    read = (c > 1e-12) & (pooled.max(1) > 0)
    zh = np.where(s[:, None, None] > 0, z / np.maximum(s, 1e-300)[:, None, None], 0)
    return s, c, zh, read


def merge(z, W, b, p, parts):
    s, c, zh, read = stats(z, W, p)
    yB = np.tile(b, (z.shape[1], 1)).astype(float)
    grouped = set(u for Q in parts for u in Q)
    for u in range(z.shape[0]):
        if u not in grouped:
            yB += np.outer(np.maximum(z[u], 0).mean(1), W[:, u])
    E = 0.0
    for Q in parts:
        R = [u for u in Q if read[u]]
        if not R:
            continue
        om = c[R] / c[R].sum()
        zQ = np.tensordot(om, zh[R], 1)
        V = (s[R][None, :] * W[:, R]).sum(1)
        yB += np.outer(np.maximum(zQ, 0).mean(1), V)
        diffs = np.sqrt((p[None, :, None] * (zh[R] - zQ[None]) ** 2).mean(2).sum(1))
        E += float((c[R] * diffs).sum())
    return yB, E


def refit(feat, yN, p, Wb0, steps=400, lr=0.5):
    """gradient descent on the convex re-fit problem; feat n x d (with a ones column)."""
    Wb = Wb0.copy()
    for _ in range(steps):
        y = feat @ Wb
        g = feat.T @ (p[:, None] * (np.exp(logsm(y)) - np.exp(logsm(yN))))
        Wb -= lr * g
    return Wb


fails = []
worst_ratio = 0.0
for it in range(N_INST):
    n, T, ny, F = rng.integers(20, 60), rng.integers(1, 6), rng.integers(2, 7), rng.integers(2, 5)
    sizes = rng.integers(1, 4, F); C = int(sizes.sum())
    base = rng.normal(size=(F, n, T)) + rng.normal(0.3, 0.5)
    fam = np.repeat(np.arange(F), sizes)
    scale = rng.uniform(0.3, 3.0, C)
    p = rng.dirichlet(np.ones(n))
    W = rng.normal(size=(ny, C)); b = rng.normal(size=ny)
    parts_f = [list(np.where(fam == f)[0]) for f in range(F)]
    for sigma in (0.0, 0.05, 0.3):
        z = scale[:, None, None] * base[fam] + sigma * rng.normal(size=(C, n, T))
        yN = b + np.maximum(z, 0).mean(2).T @ W.T
        for parts in (parts_f, [list(range(C))]):
            yB, E = merge(z, W, b, p, parts)
            D = kl(p, yN, yB)
            if sigma == 0 and parts is parts_f and D > 1e-10:
                fails.append(("A", it, D))
            if D > E ** 2 / 4 + 1e-10:
                fails.append(("B", it, sigma, D, E ** 2 / 4))
            if E > 1e-3:
                worst_ratio = max(worst_ratio, D / (E ** 2 / 4))
        # C rescaling of channel 0
        z2, W2 = z.copy(), W.copy(); z2[0] *= 2.7; W2[:, 0] /= 2.7
        yB1, E1 = merge(z, W, b, p, parts_f); yB2, E2 = merge(z2, W2, b, p, parts_f)
        if abs(E1 - E2) > 1e-9 or np.abs(yB1 - yB2).max() > 1e-9:
            fails.append(("C", it, abs(E1 - E2)))
        # D copy of channel 0 into 3
        lam = rng.dirichlet(np.ones(3))
        z3 = np.concatenate([lam[:, None, None] * z[0][None], z[1:]]); W3 = np.concatenate([np.repeat(W[:, :1], 3, 1), W[:, 1:]], 1)
        parts3 = [[0, 1, 2] + [u + 2 for u in Q if u != 0] if 0 in Q else [u + 2 for u in Q] for Q in parts_f]
        yB3, E3 = merge(z3, W3, b, p, parts3)
        if abs(E1 - E3) > 1e-9 or np.abs(yB1 - yB3).max() > 1e-9:
            fails.append(("D", it, abs(E1 - E3)))
        # E re-fit lowers the loss
        s, c, zh, read = stats(z, W, p)
        feats, V0 = [], []
        for Q in parts_f:
            R = [u for u in Q if read[u]]
            if R:
                om = c[R] / c[R].sum(); zQ = np.tensordot(om, zh[R], 1)
                feats.append(np.maximum(zQ, 0).mean(1)); V0.append((s[R][None, :] * W[:, R]).sum(1))
        if feats:
            feat = np.column_stack(feats + [np.ones(n)]); Wb0 = np.vstack(V0 + [b])
            D0 = kl(p, yN, feat @ Wb0); D1 = kl(p, yN, feat @ refit(feat, yN, p, Wb0))
            if D1 > D0 + 1e-10:
                fails.append(("E", it, D0, D1))


# F: dynamic program vs brute force on random binary trees over m leaves
def cuts(node, kids):
    if node not in kids:
        return [[node]]
    l, r = kids[node]
    out = [[node]]
    for a in cuts(l, kids):
        for bb in cuts(r, kids):
            out.append(a + bb)
    return out


for it in range(30):
    m = int(rng.integers(2, 9)); nodes = list(range(m)); kids = {}; nxt = m; pool = list(range(m))
    while len(pool) > 1:
        i, j = rng.choice(len(pool), 2, replace=False); a, bb = pool[i], pool[j]
        kids[nxt] = (a, bb); pool = [q for k, q in enumerate(pool) if k not in (i, j)] + [nxt]; nxt += 1
    root = pool[0]; cost = {q: (0.0 if q < m else float(rng.uniform(0, 1))) for q in range(nxt)}
    f = {}

    def dp(q):
        if q not in kids:
            f[q] = {1: cost[q]}; return
        l, r = kids[q]; dp(l); dp(r)
        t = {1: cost[q]}
        for k1, v1 in f[l].items():
            for k2, v2 in f[r].items():
                t[k1 + k2] = min(t.get(k1 + k2, np.inf), v1 + v2)
        f[q] = t
    dp(root)
    brute = {}
    for cut in cuts(root, kids):
        brute[len(cut)] = min(brute.get(len(cut), np.inf), sum(cost[q] for q in cut))
    if any(abs(brute[k] - f[root][k]) > 1e-12 for k in brute):
        fails.append(("F", it))

print("instances", N_INST, "x 3 noise levels; DP trees 30")
print("largest D_P / (E^2/4) over instances with E > 1e-3:", round(worst_ratio, 4))
print("failures:", len(fails), fails[:5])
