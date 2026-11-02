# serves: H-active
"""Part C, H-active: are a trained ResNet-18's areas confident in their own classes?
Estimator fixed in COARSE-THEORY.md Part C before this run (proxy areas from N alone, see the row there).
Data: cache/r18/y_train.npy (35,000 x 1000 logits, float32), uniform weights."""
import json
import os

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import squareform
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
Y = np.load(os.path.join(HERE, "cache/r18/y_train.npy")).astype(np.float64)
n, C = Y.shape
logS = Y - logsumexp(Y, axis=1, keepdims=True)
S = np.exp(logS)
top = S.argmax(1)

# class similarity C_ij = mean of s_j over photos with top class i, normalised so C_ii = 1
M = np.zeros((C, C))
cnt = np.bincount(top, minlength=C)
np.add.at(M, top, S)
M = M / np.maximum(cnt, 1)[:, None]
d = np.diag(M).copy()
d[d == 0] = 1
M = M / d[:, None]
sim = (M + M.T) / 2
np.fill_diagonal(sim, 1)
D = np.clip(1 - sim, 0, None)
np.fill_diagonal(D, 0)
Z = linkage(squareform(D, checks=False), method="average")


def phi(t):
    return np.expm1(t) - t


def eps_act(Ya, Sa, K):
    """Definition DQ1 on one area: Ya centred logits (P_perp y minus area mean), Sa softmax, K boolean mask."""
    w = 1.0 / n
    rho = np.linalg.norm(Ya, axis=1)                      # |Y(x)|_2
    out = ~K
    Delta = np.abs(Ya[:, out]) + rho[:, None]             # Delta_(x,i), i not in K
    so = Sa[:, out]

    def f(b):
        return w * np.sum(so * np.maximum(phi(Delta - b), phi(-Delta - b)))
    dmax = Delta.max()
    r = minimize_scalar(f, bounds=(-1.0, dmax + 1.0), method="bounded", options={"xatol": 1e-6})
    return float(r.fun), float(r.x), float(dmax), float(w * so.sum()), float(np.median(rho))


def eps_act2(Ya, Sa, K, iters=80):
    """Definition DQ1 (version 2): signed own part, envelope only rho_K = |Pi_K Y(x)|_2, one shift beta_i per class.
    term_i(beta) = sum_x p s_(x,i) max(phi(-Y_(x,i) - beta + rho_K(x)), phi(-Y_(x,i) - beta - rho_K(x))); convex in
    beta; minimised per class by bisection on the slope."""
    w = 1.0 / n
    rhoK = np.linalg.norm(Ya[:, K], axis=1)[:, None]
    out = ~K
    U = -Ya[:, out]
    so = Sa[:, out]
    lo = np.full(out.sum(), -50.0)
    hi = np.full(out.sum(), 300.0)

    def slope(beta):
        a1 = U - beta + rhoK
        a2 = U - beta - rhoK
        f1, f2 = phi(a1), phi(a2)
        g = np.where(f1 >= f2, -np.expm1(a1), -np.expm1(a2))   # d/dbeta of phi(u - beta +- rho)
        return (so * g).sum(0)

    for _ in range(iters):
        mid = (lo + hi) / 2
        sl = slope(mid)
        lo = np.where(sl < 0, mid, lo)
        hi = np.where(sl < 0, hi, mid)
    beta = (lo + hi) / 2
    a1 = U - beta + rhoK
    a2 = U - beta - rhoK
    val = w * float((so * np.maximum(phi(a1), phi(a2))).sum())
    return val, float(np.median(rhoK))


def info(Sa, logSa):
    w = 1.0 / n
    sb = Sa.mean(0)
    return float(w * np.sum(Sa * (logSa - np.log(sb))))


res = {}
for G in [10, 30, 100, 300]:
    lab = fcluster(Z, t=G, criterion="maxclust") - 1
    G_eff = lab.max() + 1
    rows = []
    for g in range(G_eff):
        K = lab == g
        idx = np.where(K[top])[0]
        if len(idx) < 2:
            continue
        P = Y[idx] - Y[idx].mean(1, keepdims=True)         # P_perp y
        Ya = P - P.mean(0)
        e, b, dmax, mu, rmed = eps_act(Ya, S[idx], K)
        I = info(S[idx], logS[idx])
        e2, rK = eps_act2(Ya, S[idx], K)
        rows.append(dict(eps_act2=e2, rhoK_median=rK, g=int(g), n_area=len(idx), n_classes=int(K.sum()), mu=mu, eps_act=e, b=b, dmax=dmax,
                         rho_median=rmed, I=I, bound_closed=2 * dmax * mu))
    E = sum(r["eps_act"] for r in rows)
    I = sum(r["I"] for r in rows)
    ratios = [r["eps_act"] / r["I"] for r in rows if r["I"] > 0]
    E2 = sum(r["eps_act2"] for r in rows)
    ratios2 = [r["eps_act2"] / r["I"] for r in rows if r["I"] > 0]
    res[G] = dict(blocks=int(G_eff), areas=len(rows), sum_eps_act=E, sum_I=I, ratio=E / I,
                  median_ratio=float(np.median(ratios)), mean_outside_mass=sum(r["mu"] for r in rows),
                  median_rho=float(np.median([r["rho_median"] for r in rows])),
                  median_dmax=float(np.median([r["dmax"] for r in rows])),
                  sum_closed=sum(r["bound_closed"] for r in rows), sum_eps_act2=E2, ratio2=E2 / I,
                  median_ratio2=float(np.median(ratios2)),
                  median_rhoK=float(np.median([r["rhoK_median"] for r in rows])), rows=rows)
    print(f"G={G:4d} areas={len(rows):4d} outside mass={res[G]['mean_outside_mass']:.3f} "
          f"sum eps_act={E:.3f} sum I={I:.3f} ratio={E / I:.3f} median ratio={res[G]['median_ratio']:.3f} "
          f"median rho={res[G]['median_rho']:.2f} median Dmax={res[G]['median_dmax']:.2f} closed={res[G]['sum_closed']:.2f}")
    print(f"        v2: sum eps_act2={E2:.3f} ratio2={E2 / I:.3f} median ratio2={res[G]['median_ratio2']:.3f} "
          f"median rho_K={res[G]['median_rhoK']:.2f}")

json.dump({str(k): v for k, v in res.items()}, open(os.path.join(HERE, "results_pc_active.json"), "w"), indent=1)


# Part 2 (added after the first run, same proxy areas): the exact loss of the packed choice of Theorem TQ1 part 2,
# not its bound. k = the largest own-class rank over the areas (complete case), canonical SVD bases, one output
# direction per column shared by all areas, per-area offset t_w fitted (convex). Compared with sum I (drop all detail).
from scipy.optimize import minimize as _min


def packed_loss(G, kcap=None):
    lab = fcluster(Z, t=G, criterion="maxclust") - 1
    areas = []
    for g in range(lab.max() + 1):
        K = lab == g
        idx = np.where(K[top])[0]
        if len(idx) < 2:
            continue
        P = Y[idx] - Y[idx].mean(1, keepdims=True)
        Ya = P - P.mean(0)
        YK = Ya[:, K]
        u, sv, vt = np.linalg.svd(YK, full_matrices=False)
        r = int((sv > 1e-8 * max(sv[0], 1e-12)).sum())
        areas.append(dict(K=K, idx=idx, Ya=Ya, ybar=P.mean(0), vt=vt[:r], r=r))
    k = max(a["r"] for a in areas) if kcap is None else kcap
    V = np.zeros((C, k))
    for a in areas:
        rr = min(a["r"], k)
        cols = np.where(a["K"])[0]
        V[cols, :rr] += a["vt"][:rr].T                      # Pi_w U_w
    tot, totI = 0.0, 0.0
    for a in areas:
        rr = min(a["r"], k)
        cw = a["Ya"][:, a["K"]] @ a["vt"][:rr].T            # U_w^T Pi_w Y_w
        L0 = cw @ V[:, :rr].T                               # V c_w(x)
        s = S[a["idx"]]
        ls = logS[a["idx"]]

        def f(t):
            L = L0 + t
            lq = L - logsumexp(L, axis=1, keepdims=True)
            val = np.sum(s * (ls - lq)) / n
            g = (np.exp(lq) - s).sum(0) / n
            return val, g
        t0 = a["ybar"] - (L0.mean(0))
        res_ = _min(f, t0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
        tot += res_.fun
        totI += info(s, ls)
    return k, tot, totI


for G in [100, 300]:
    for kc in [None, 8, 2]:
        k, L, I = packed_loss(G, kc)
        tag = "complete (k = max own rank)" if kc is None else f"k = {kc} (areas above k truncated: not complete)"
        print(f"G={G} {tag}: k={k} packed loss={L:.3f} sum I={I:.3f} ratio={L / I:.3f}")
        res[f"packed_{G}_{kc}"] = dict(k=k, packed_loss=L, sum_I=I, ratio=L / I)
json.dump({str(k): v for k, v in res.items()}, open(os.path.join(HERE, "results_pc_active.json"), "w"), indent=1)
