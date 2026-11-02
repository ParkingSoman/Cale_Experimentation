# serves: H-active
"""Part C, H-active, per-area scales (estimator in COARSE-THEORY.md Part C, fixed before the run).
Canonical packed choice (pc_active.py part 2) plus one scale lambda_w per area; offsets t_w fitted; directions and
reads fixed. Run with /opt/homebrew/bin/python3.11 (torch)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(HERE, "cache/r18/active_labels.npz")
Y = np.load(os.path.join(HERE, "cache/r18/y_train.npy")).astype(np.float64)
n, C = Y.shape
logS = Y - Y.max(1, keepdims=True)
logS = logS - np.log(np.exp(logS).sum(1, keepdims=True))
S = np.exp(logS)
top = S.argmax(1)

if "--labels" in sys.argv:   # stage 1 (default python3, scipy): the proxy areas of pc_active.py
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    M = np.zeros((C, C)); cnt = np.bincount(top, minlength=C)
    np.add.at(M, top, S); M = M / np.maximum(cnt, 1)[:, None]
    d = np.diag(M).copy(); d[d == 0] = 1; M = M / d[:, None]
    sim = (M + M.T) / 2; np.fill_diagonal(sim, 1)
    D = np.clip(1 - sim, 0, None); np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method="average")
    np.savez(LABELS, **{f"G{G}": fcluster(Z, t=G, criterion="maxclust") - 1 for G in [100, 300]})
    sys.exit(0)

import torch  # noqa: E402  (stage 2: python3.11 with torch)
torch.set_default_dtype(torch.float64)
LAB = np.load(LABELS)


def run(G, kcap=None, iters=300):
    lab = LAB[f"G{G}"]
    areas = []
    for g in range(lab.max() + 1):
        K = np.where(lab == g)[0]
        idx = np.where(np.isin(top, K))[0]
        if len(idx) < 2:
            continue
        P = Y[idx] - Y[idx].mean(1, keepdims=True)
        Ya = P - P.mean(0)
        X = Ya[:, K]
        u, sv, vt = np.linalg.svd(X, full_matrices=False)
        r = int((sv > 1e-8 * max(sv[0], 1e-12)).sum())
        I = float(np.sum(S[idx] * (logS[idx] - np.log(S[idx].mean(0)))) / n)
        areas.append(dict(K=K, idx=idx, X=X, ybar=P.mean(0), vt=vt[:r], r=r, I=I))
    k = max(a["r"] for a in areas) if kcap is None else kcap
    U = np.zeros((len(areas), C, k))
    cs = []
    for j, a in enumerate(areas):
        rr = min(a["r"], k)
        U[j][a["K"], :rr] = a["vt"][:rr].T
        c = np.zeros((len(a["idx"]), k)); c[:, :rr] = a["X"] @ a["vt"][:rr].T
        cs.append(torch.tensor(c))
    Ut = torch.tensor(U)
    Ss = [torch.tensor(S[a["idx"]]) for a in areas]
    lS = [torch.tensor(logS[a["idx"]]) for a in areas]
    loglam = torch.zeros(len(areas), requires_grad=True)
    T = torch.tensor(np.stack([a["ybar"] for a in areas]), requires_grad=True)

    def loss(use_scale=True):
        lam = torch.exp(loglam) if use_scale else torch.ones(len(areas))
        V = (lam[:, None, None] * Ut).sum(0)            # C x k
        tot = 0.0
        for j in range(len(areas)):
            L = (cs[j] / lam[j]) @ V.T + T[j]
            lq = L - torch.logsumexp(L, dim=1, keepdim=True)
            tot = tot + (Ss[j] * (lS[j] - lq)).sum() / n
        return tot

    # offsets only (canonical packed choice, fitted offsets)
    opt = torch.optim.LBFGS([T], max_iter=iters, line_search_fn="strong_wolfe")

    def cl0():
        opt.zero_grad(); v = loss(False); v.backward(); return v
    opt.step(cl0)
    base = float(loss(False))
    # scales and offsets
    opt2 = torch.optim.LBFGS([loglam, T], max_iter=iters, line_search_fn="strong_wolfe")

    def cl1():
        opt2.zero_grad(); v = loss(True); v.backward(); return v
    opt2.step(cl1)
    scaled = float(loss(True))
    I = sum(a["I"] for a in areas)
    out = dict(G=G, k=k, canonical=base, scaled=scaled, sum_I=I, canonical_ratio=base / I, scaled_ratio=scaled / I,
               lam_spread=float(torch.exp(loglam).max() / torch.exp(loglam).min()))
    print(out, flush=True)
    return out


def run_gauge(G, kcap=None, iters=300):
    """per-area k x k gauge A_w (Conjecture DQ2): V = sum Pi U_w A_w, area w reads A_w^{-1} c_w; offsets fitted."""
    lab = LAB[f"G{G}"]
    areas = []
    for g in range(lab.max() + 1):
        K = np.where(lab == g)[0]
        idx = np.where(np.isin(top, K))[0]
        if len(idx) < 2:
            continue
        P = Y[idx] - Y[idx].mean(1, keepdims=True)
        Ya = P - P.mean(0)
        X = Ya[:, K]
        u, sv, vt = np.linalg.svd(X, full_matrices=False)
        r = int((sv > 1e-8 * max(sv[0], 1e-12)).sum())
        I = float(np.sum(S[idx] * (logS[idx] - np.log(S[idx].mean(0)))) / n)
        areas.append(dict(K=K, idx=idx, X=X, ybar=P.mean(0), vt=vt[:r], r=r, I=I))
    k = max(a["r"] for a in areas) if kcap is None else kcap
    U = np.zeros((len(areas), C, k)); cs = []
    for j, a in enumerate(areas):
        rr = min(a["r"], k)
        U[j][a["K"], :rr] = a["vt"][:rr].T
        c = np.zeros((len(a["idx"]), k)); c[:, :rr] = a["X"] @ a["vt"][:rr].T
        cs.append(torch.tensor(c))
    Ut = torch.tensor(U)
    Ss = [torch.tensor(S[a["idx"]]) for a in areas]
    lS = [torch.tensor(logS[a["idx"]]) for a in areas]
    A = torch.eye(k).repeat(len(areas), 1, 1).clone().requires_grad_(True)
    T = torch.tensor(np.stack([a["ybar"] for a in areas]), requires_grad=True)

    def loss():
        V = torch.einsum("wck,wkj->cj", Ut, A)
        tot = 0.0
        for j in range(len(areas)):
            cj = torch.linalg.solve(A[j], cs[j].T).T          # A_w^{-1} c_w
            L = cj @ V.T + T[j]
            lq = L - torch.logsumexp(L, dim=1, keepdim=True)
            tot = tot + (Ss[j] * (lS[j] - lq)).sum() / n
        return tot
    opt0 = torch.optim.LBFGS([T], max_iter=iters, line_search_fn="strong_wolfe")

    def c0():
        opt0.zero_grad(); v = loss(); v.backward(); return v
    opt0.step(c0)
    base = float(loss())
    opt = torch.optim.LBFGS([A, T], max_iter=iters, line_search_fn="strong_wolfe")

    def c1():
        opt.zero_grad(); v = loss(); v.backward(); return v
    opt.step(c1)
    g = float(loss())
    I = sum(a["I"] for a in areas)
    out = dict(G=G, k=k, canonical=base, gauge=g, sum_I=I, canonical_ratio=base / I, gauge_ratio=g / I)
    print(out, flush=True)
    return out


if "--gauge" in sys.argv:
    res = [run_gauge(300, 2), run_gauge(300), run_gauge(100)]
    json.dump(res, open(os.path.join(HERE, "results_pc_active_gauge.json"), "w"), indent=1)
else:
    res = [run(100), run(300), run(300, 2)]
    json.dump(res, open(os.path.join(HERE, "results_pc_active_scale.json"), "w"), indent=1)
