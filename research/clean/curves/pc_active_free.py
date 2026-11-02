# serves: H-active
"""Part C, H-active, two read-outs (estimator in COARSE-THEORY.md Part C, fixed before the run).
Same proxy areas as pc_active_scale.py (cache/r18/active_labels.npz), complete cases only (k = max own rank).
(1) Pinning: the gauge solution (own SVD directions U_w fixed, one k x k map A_w per area, offsets) run to
    convergence; then a head with V free (C x k), reads R_w free, offsets free, started AT the gauge solution and run
    to convergence. If the free head gains nothing above noise, the own directions are pinned on this network.
(2) Structure of the gauges: for each ordered pair (w, w') the map T = A_w' A_w^-1 from w's own coordinates to w''s
    (the leak of area w onto block K_w'). Reported for the pairs with the largest leak weight (mass of K_w' on A_w):
    condition number of T, share of |T|^2 on a greedy permutation (a lower bound on the best one), and whether that permutation is the reverse
    order (TQ4) in the own directions sorted by strength.
(3) Distance from the confident limit of Theorem TQ5: the gauge build is a leak point; by TQ5 part 2 at m = 0 its KL
    is at most Lambda of that point (odds rho); KL / Lambda tends to 1 in the limit.
Run with /opt/homebrew/bin/python3.11 (torch)."""
import json
import os
import sys
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
Y = np.load(os.path.join(HERE, "cache/r18/y_train.npy")).astype(np.float64)
LAB = np.load(os.path.join(HERE, "cache/r18/active_labels.npz"))
n, C = Y.shape
logS = Y - Y.max(1, keepdims=True)
logS = logS - np.log(np.exp(logS).sum(1, keepdims=True))
S = np.exp(logS)
top = S.argmax(1)


def setup(G):
    lab = LAB[f"G{G}"]
    areas = []
    for g in range(lab.max() + 1):
        K = np.where(lab == g)[0]
        idx = np.where(np.isin(top, K))[0]
        if len(idx) < 2:
            continue
        P = Y[idx] - Y[idx].mean(1, keepdims=True)
        X = (P - P.mean(0))[:, K]
        u, sv, vt = np.linalg.svd(X, full_matrices=False)
        r = int((sv > 1e-8 * max(sv[0], 1e-12)).sum())
        I = float(np.sum(S[idx] * (logS[idx] - np.log(S[idx].mean(0)))) / n)
        areas.append(dict(K=K, idx=idx, X=X, ybar=P.mean(0), vt=vt[:r], sv=sv[:r], r=r, I=I))
    return areas


def lbfgs(params, f, rounds=5, iters=100, tol=1e-9):
    opt = torch.optim.LBFGS(params, max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-12,
                            tolerance_change=1e-14, history_size=50)
    prev = None
    for _ in range(rounds):
        def cl():
            opt.zero_grad(); v = f(); v.backward(); return v
        opt.step(cl)
        v = float(f())
        if prev is not None and prev - v < tol:
            break
        prev = v
    return float(f())


def run(G):
    t0 = time.time()
    areas = setup(G)
    k = max(a["r"] for a in areas)
    W = len(areas)
    U = np.zeros((W, C, k)); cs = []
    for j, a in enumerate(areas):
        U[j][a["K"], :a["r"]] = a["vt"].T
        c = np.zeros((len(a["idx"]), k)); c[:, :a["r"]] = a["X"] @ a["vt"].T
        cs.append(torch.tensor(c))
    Ut = torch.tensor(U)
    Ss = [torch.tensor(S[a["idx"]]) for a in areas]
    lS = [torch.tensor(logS[a["idx"]]) for a in areas]
    A = torch.eye(k).repeat(W, 1, 1).clone().requires_grad_(True)
    T = torch.tensor(np.stack([a["ybar"] for a in areas]), requires_grad=True)

    def kl(Ls):
        tot = 0.0
        for j, L in enumerate(Ls):
            lq = L - torch.logsumexp(L, dim=1, keepdim=True)
            tot = tot + (Ss[j] * (lS[j] - lq)).sum() / n
        return tot

    def loss_gauge():
        V = torch.einsum("wck,wkj->cj", Ut, A)
        return kl([torch.linalg.solve(A[j], cs[j].T).T @ V.T + T[j] for j in range(W)])
    lbfgs([T], loss_gauge, rounds=2)
    g = lbfgs([A, T], loss_gauge)
    # (3) distance from the confident limit (Theorem TQ5 part 2 at m = 0): the gauge build is a leak point; its KL is
    # at most Lambda of that point (odds rho, its own beta); the ratio KL / Lambda is 1 in the limit
    with torch.no_grad():
        V = torch.einsum("wck,wkj->cj", Ut, A)
        klsum = 0.0; lamsum = 0.0
        for j, a in enumerate(areas):
            l = (torch.linalg.solve(A[j], cs[j].T).T @ V.T + T[j]).numpy()
            y = Y[a["idx"]]; Kj = a["K"]; out = np.setdiff1d(np.arange(C), Kj)
            lseK_l = np.logaddexp.reduce(l[:, Kj], axis=1, keepdims=True)
            lseK_y = np.logaddexp.reduce(y[:, Kj], axis=1, keepdims=True)
            dd = (l[:, out] - lseK_l) - (y[:, out] - lseK_y)
            rh = np.exp(y[:, out] - lseK_y)
            lamsum += float((rh * (np.expm1(dd) - dd)).sum() / n)
            lq = l - np.logaddexp.reduce(l, axis=1, keepdims=True)
            klsum += float((S[a["idx"]] * (logS[a["idx"]] - lq)).sum() / n)
    # free head started at the gauge solution
    with torch.no_grad():
        V0 = torch.einsum("wck,wkj->cj", Ut, A).clone()
        R0 = torch.stack([torch.linalg.inv(A[j]) for j in range(W)]).clone()
    V = V0.clone().requires_grad_(True)
    R = R0.clone().requires_grad_(True)
    T2 = T.detach().clone().requires_grad_(True)

    def loss_free():
        return kl([cs[j] @ R[j].T @ V.T + T2[j] for j in range(W)])
    f0 = float(loss_free())
    fr = lbfgs([V, R, T2], loss_free)
    sumI = sum(a["I"] for a in areas)
    # structure of the gauges
    Ad = A.detach().numpy()
    pairs = []
    for w, a in enumerate(areas):
        mass = S[a["idx"]].sum(0) / n
        for w2, b in enumerate(areas):
            if w2 == w:
                continue
            pairs.append((float(mass[b["K"]].sum()), w, w2))
    pairs.sort(reverse=True)
    rows = []
    for m, w, w2 in pairs[:200]:
        rw, rw2 = areas[w]["r"], areas[w2]["r"]
        Tm = (Ad[w2] @ np.linalg.inv(Ad[w]))[:rw2, :rw]
        sv = np.linalg.svd(Tm, compute_uv=False)
        E = Tm ** 2
        Eg = E.copy(); got = 0.0                     # greedy permutation: a lower bound on the best one
        for _ in range(min(Eg.shape)):
            i, j = np.unravel_index(np.argmax(Eg), Eg.shape); got += E[i, j]; Eg[i, :] = -1; Eg[:, j] = -1
        perm_share = float(got / E.sum())
        m_min = min(rw, rw2)
        rev = np.array([(i, m_min - 1 - i) for i in range(m_min)])
        rev_share = float(E[rev[:, 0], rev[:, 1]].sum() / E.sum())
        diag_share = float(sum(E[i, i] for i in range(m_min)) / E.sum())
        rows.append(dict(mass=m, cond=float(sv[0] / max(sv[-1], 1e-300)), perm=perm_share, rev=rev_share,
                         diag=diag_share))
    def med(key):
        return float(np.median([r[key] for r in rows]))
    out = dict(G=G, k=k, areas=W, sum_I=sumI, gauge=g / sumI, tq5_kl_over_lambda=klsum / lamsum, free_start=f0 / sumI, free=fr / sumI,
               free_gain_rel=(g - fr) / g, pairs=len(rows), med_cond=med("cond"), med_perm_share=med("perm"),
               med_rev_share=med("rev"), med_diag_share=med("diag"),
               random_perm_share_ref=1.0 / k, seconds=time.time() - t0)
    print(out, flush=True)
    return out


Gs = [int(a) for a in sys.argv[1:]] or [100]
res = [run(G) for G in Gs]
json.dump(res, open(os.path.join(HERE, "results_pc_active_free.json"), "w"), indent=1)
