# serves: H-active
"""Part C, H-active discriminator (estimator in COARSE-THEORY.md Part C, fixed before the run).
Same proxy areas as pc_active.py. Head: logits on area w = t_w + V R_w Pi_w Y_w(x); V shared (C x k), R_w (k x |K_w|),
t_w (C). Start at the packed choice of TQ1 part 2; L-BFGS on the summed KL. Local, uncertified."""
import json
import os
import time

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
Y = np.load(os.path.join(HERE, "cache/r18/y_train.npy")).astype(np.float64)
n, C = Y.shape
logS = Y - logsumexp(Y, axis=1, keepdims=True)
S = np.exp(logS)
top = S.argmax(1)
M = np.zeros((C, C)); cnt = np.bincount(top, minlength=C)
np.add.at(M, top, S); M = M / np.maximum(cnt, 1)[:, None]
d = np.diag(M).copy(); d[d == 0] = 1; M = M / d[:, None]
sim = (M + M.T) / 2; np.fill_diagonal(sim, 1)
D = np.clip(1 - sim, 0, None); np.fill_diagonal(D, 0)
Z = linkage(squareform(D, checks=False), method="average")


def setup(G):
    lab = fcluster(Z, t=G, criterion="maxclust") - 1
    areas = []
    for g in range(lab.max() + 1):
        K = np.where(lab == g)[0]
        idx = np.where(np.isin(top, K))[0]
        if len(idx) < 2:
            continue
        P = Y[idx] - Y[idx].mean(1, keepdims=True)
        Ya = P - P.mean(0)
        X = Ya[:, K]                                   # Pi_w Y_w on own classes
        u, sv, vt = np.linalg.svd(X, full_matrices=False)
        r = int((sv > 1e-8 * max(sv[0], 1e-12)).sum())
        I = float(np.sum(S[idx] * (logS[idx] - np.log(S[idx].mean(0)))) / n)
        areas.append(dict(K=K, idx=idx, X=X, ybar=P.mean(0), vt=vt[:r], r=r, I=I))
    return areas


def run(G, kcap=None, maxiter=300):
    areas = setup(G)
    k = max(a["r"] for a in areas) if kcap is None else kcap
    V0 = np.zeros((C, k)); R0 = []
    for a in areas:
        rr = min(a["r"], k)
        R = np.zeros((k, len(a["K"]))); R[:rr] = a["vt"][:rr]
        V0[a["K"], :rr] += a["vt"][:rr].T
        R0.append(R)
    T0 = [a["ybar"] - (a["X"] @ R.T @ V0.T).mean(0) for a, R in zip(areas, R0)]
    shapes = [V0.shape] + [R.shape for R in R0] + [t.shape for t in T0]
    sizes = [int(np.prod(sh)) for sh in shapes]

    def unpack(v):
        out, o = [], 0
        for sh, sz in zip(shapes, sizes):
            out.append(v[o:o + sz].reshape(sh)); o += sz
        return out[0], out[1:1 + len(areas)], out[1 + len(areas):]

    def f(v):
        V, Rs, Ts = unpack(v)
        val = 0.0; gV = np.zeros_like(V); gR = []; gT = []
        for a, R, t in zip(areas, Rs, Ts):
            c = a["X"] @ R.T                           # (m, k)
            L = c @ V.T + t
            lq = L - logsumexp(L, axis=1, keepdims=True)
            s = S[a["idx"]]
            val += np.sum(s * (logS[a["idx"]] - lq)) / n
            dL = (np.exp(lq) - s) / n
            gT.append(dL.sum(0))
            gV += dL.T @ c
            dc = dL @ V
            gR.append(dc.T @ a["X"])
        return val, np.concatenate([gV.ravel()] + [g.ravel() for g in gR] + [g.ravel() for g in gT])

    v0 = np.concatenate([V0.ravel()] + [R.ravel() for R in R0] + [t.ravel() for t in T0])
    f0 = f(v0)[0]
    t1 = time.time()
    r = minimize(f, v0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    I = sum(a["I"] for a in areas)
    out = dict(G=G, k=k, packed=f0, fitted=float(r.fun), sum_I=I, packed_ratio=f0 / I, fitted_ratio=float(r.fun) / I,
               iters=int(r.nit), seconds=time.time() - t1)
    print(out, flush=True)
    return out


res = []
for G, kc in [(300, None), (300, 2), (100, None), (100, 2)]:
    res.append(run(G, kc))
json.dump(res, open(os.path.join(HERE, "results_pc_active_opt.json"), "w"), indent=1)
