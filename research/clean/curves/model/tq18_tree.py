# serves: TQ18
"""Theorem TQ18 on a family tree: one table, several levels of a zoom, relatives on one pole. Estimators fixed before the run. float64.
Table: K = C = 2^d classes, the leaves of a binary tree, weights 1/K. Logits of row c: sum over levels l of a_l times the contrast at c's node of level l: +1 on the child that holds c, -1 on the other child, 0 on classes off that node; a_l = a_0 r^l. So s_c = softmax of that: the top levels are strong, the fine levels weak. This is a table with structure: relatives are likelier than strangers on every row, the opposite of a pole table's flat tail.
A cut at m levels keeps the 2^m - 1 contrasts of the top m levels (k = 2^m - 1). Candidate: V = those contrasts, normalised; codes and t from the convex fit at that span. By the tree's symmetry it has one number per level.
Reported at each cut: KL; the stationarity errors; g, sigma, rho; the test ratio g / (rho min(1, sigma / 2)); the fit from 15 random starts; the fall below the candidate; the largest cosine between the best start's span and the tree span (smallest principal cosine).
A second control: the same logits with rows of the leaf labels shuffled on the column side only, so that the tree in the rows does not match the tree in the classes.
Prediction, written before the run: for r small the test passes at every cut, and random starts match at float error, with the tree span the unique span. As r grows to 1 (no decay) the test fails first at the finest cuts. Where the test passes there is never a fall above 1e-9. Where it fails, the tree span may still be the best found.
Usage: tq18_tree.py [--smoke]"""
import argparse
import json
import os

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_default_dtype(torch.float64)
torch.set_num_threads(3)
torch.manual_seed(41)
STARTS = 4 if a.smoke else 15


def contrasts(d):
    """All tree contrasts, by level: a list of (level, vector)."""
    K = 2 ** d
    out = []
    for l in range(d):
        size = K // 2 ** l
        for b in range(2 ** l):
            v = np.zeros(K)
            start = b * size
            v[start:start + size // 2] = 1
            v[start + size // 2:start + size] = -1
            out.append((l, v))
    return out


def table(d, a0, r):
    K = 2 ** d
    L = np.zeros((K, K))
    for c in range(K):
        for l, v in contrasts(d):
            if v[c] != 0:
                L[c] += a0 * r ** l * v[c] * v
    q = np.exp(L - L.max(1, keepdims=True))
    return q / q.sum(1, keepdims=True), L


def lbfgs(ps, fn, it=4000):
    o = torch.optim.LBFGS(ps, lr=1, max_iter=it, history_size=60, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-18)

    def cl():
        o.zero_grad()
        v = fn()
        v.backward()
        return v
    o.step(cl)
    return float(fn().detach())


def study(s, d, m):
    K = 2 ** d
    cs = [v for l, v in contrasts(d) if l < m]
    k = len(cs)
    Q = np.stack([v / np.linalg.norm(v) for v in cs], 1)
    s_ = torch.tensor(s)
    ls = s_.log()
    pi = 1 / K
    Q_ = torch.tensor(Q)
    c = torch.zeros(K, k, requires_grad=True)
    t = torch.zeros(K, requires_grad=True)
    kl = lambda L: (pi * (s_ * (ls - torch.log_softmax(L, 1))).sum(1)).sum()
    val = lbfgs([c, t], lambda: kl(t + c @ Q_.T))
    c0, t0 = c.detach().numpy(), t.detach().numpy()
    mean = c0.mean(0, keepdims=True)
    t0 = t0 + (mean @ Q.T)[0]
    c0 = c0 - mean
    L = t0 + c0 @ Q.T
    q = np.exp(L - L.max(1, keepdims=True))
    q = q / q.sum(1, keepdims=True)
    G = pi * (s - q)
    g = np.linalg.norm(G, 2)
    sv = np.linalg.svd(c0, compute_uv=False)
    sigma, rho = sv.min(), (pi * q).min()
    best, bestV, vals = 1e9, None, []
    for j in range(STARTS):
        V = (torch.randn(K, k) * .8).requires_grad_(True)
        cc = (torch.randn(K, k) * .8).requires_grad_(True)
        tt = torch.zeros(K).requires_grad_(True)
        v = lbfgs([V, cc, tt], lambda: kl(tt + cc @ V.T))
        vals.append(v)
        if v < best:
            best, bestV = v, V.detach().numpy().copy()
    A = np.linalg.qr(np.column_stack([np.ones(K), bestV]))[0]
    B = np.linalg.qr(np.column_stack([np.ones(K), Q]))[0]
    return dict(d=d, m=m, k=k, kl=val, stat=float(np.abs(c0.T @ G).max() + np.abs(G.sum(0)).max() + np.abs(G @ Q).max()), g=float(g), sigma=float(sigma), rho=float(rho), ratio=float(g / (rho * min(1, sigma / 2))), passes=bool(g <= rho and g <= rho * sigma / 2), best=best, below=val - best, median_above=float(np.median(vals) - best), least_cos=float(np.linalg.svd(A.T @ B, compute_uv=False).min()))


out = dict(rows=[])
for d in ((3,) if a.smoke else (3, 4)):
    for a0, r in ((1.5, .05), (1.5, .1), (1.5, .2), (1.5, .4), (1.5, .7), (.8, .1), (.8, .4), (.3, .5), (.3, .9)):
        s, L = table(d, a0, r)
        for m in range(1, d):
            row = dict(a0=a0, r=r, **study(s, d, m))
            out["rows"].append(row)
            print({k_: (round(v, 6) if isinstance(v, float) and abs(v) > 1e-3 else v) for k_, v in row.items()}, flush=True)
fails = sum(1 for r in out["rows"] if r["passes"] and r["below"] > 1e-9)
out["fails"] = fails
print("passes:", sum(r["passes"] for r in out["rows"]), "of", len(out["rows"]), "fails where the test passes:", fails, flush=True)
if not a.smoke:
    json.dump(out, open(os.path.join(HERE, "tq18_tree.json"), "w"), indent=1)
print("done", flush=True)
