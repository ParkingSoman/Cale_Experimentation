# serves: TQ3
"""Check of Theorem TQ3 (COARSE-THEORY.md B.Q): the clean-limit sandwich, always-on boxes, k = 1.
5 classes. Area 1: K1 = {0,1,2}, 4 points z in {(+-1,0),(0,+-1)}, logits on K1 = a F z (F 3x2, centred rows: own
rank 2 > k); area 2: K2 = {3,4}, 4 points c in {-1.5,-.5,.5,1.5}, logits on K2 = a c (e3-e4)/sqrt2 (own rank 1).
Outside classes of each area sit at -m (+ a fixed small variation g z or g c, so the outside logits also move).
Weights 1/8. For m in {0, 1, 2, 4, 6, 8} (and an uneven family: -m on area 1, -2m on area 2):
  Lc1 = own-block least of area 1 (3-class problem, k = 1, on-boxes: min over lines in the block of a convex fit)
  Lc2 = own-block least of area 2 (= 0: rank 1)
  Lon = group least over on-boxes: min over output directions v (sphere search + refine) of the per-area convex fits
  glued = the glued build of TQ3 part 2 (its exact loss) and its bound sum (Lc + E)
Fail if: lower side (1 - mu/p) Lc > Lon + 1e-7, or glued loss > bound + 1e-9, or Lon > glued + 1e-7 (search worse
than an explicit build). Reported: Lon / (Lc1 + Lc2) -> 1 as m grows."""
import json
import os

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(3134)
C = 5
Z1 = np.array([[1., 0], [-1, 0], [0, 1], [0, -1]])
c2 = np.array([-1.5, -.5, .5, 1.5])
p = np.full(4, 1 / 8)
F = np.array([[1., 0.3], [-0.5, 1.0], [-0.5, -1.3]])
F -= F.mean(0)
a, g = 1.5, 0.3
K1, K2 = np.array([1, 1, 1, 0, 0], bool), np.array([0, 0, 0, 1, 1], bool)
r2 = np.array([1., -1]) / np.sqrt(2)
Bc = np.linalg.qr(np.eye(C) - 1 / C)[0][:, :C - 1]
fails = []


def logits(m1, m2):
    Y1 = np.zeros((4, C)); Y1[:, K1] = a * Z1 @ F.T
    Y1[:, 3] = -m1 + g * Z1[:, 0]; Y1[:, 4] = -m1 - g * Z1[:, 1]
    Y2 = np.zeros((4, C)); Y2[:, K2] = a * np.outer(c2, r2)
    Y2[:, 0] = -m2 + g * c2; Y2[:, 1] = -m2; Y2[:, 2] = -m2 - g * c2
    return Y1, Y2


def lsm(L):
    return L - logsumexp(L, axis=1, keepdims=True)


def fit(S, feats, V):
    """least over t (dim = rows) and read B (k x f) of sum p KL(S || softmax(t + feats B^T V^T)); convex."""
    n, Cc = S.shape
    k, f = V.shape[1], feats.shape[1]

    def obj(v):
        t, B = v[:Cc], v[Cc:].reshape(k, f)
        L = t + feats @ B.T @ V.T
        lq = lsm(L)
        gL = p[:, None] * (np.exp(lq) - S)
        val = float(np.sum(p * np.sum(S * (np.log(S) - lq), axis=1)))
        gB = V.T @ gL.T @ feats
        return val, np.concatenate([gL.sum(0), gB.ravel()])
    v0 = np.concatenate([np.log(np.average(S, axis=0, weights=p)), np.zeros(k * f)])
    r = minimize(obj, v0, jac=True, method="L-BFGS-B", options={"maxiter": 5000, "gtol": 1e-12, "ftol": 1e-16})
    t, B = r.x[:Cc], r.x[Cc:].reshape(k, f)
    return r.fun, t, B


def own_block_least(Y, K, feats, ngrid=720):
    """k = 1 own-block least on block K: min over unit lines u in the centred block space."""
    S = np.exp(lsm(Y[:, K]))
    nk = K.sum()
    Bk = np.linalg.qr(np.eye(nk) - 1 / nk)[0][:, :nk - 1]
    if nk - 1 == 1:
        val, t, B = fit(S, feats, Bk)
        return val, Bk[:, 0], t, B
    best = (np.inf, None)
    for th in np.linspace(0, np.pi, ngrid, endpoint=False):
        u = Bk @ np.array([np.cos(th), np.sin(th)])
        val = fit(S, feats, u[:, None])[0]
        if val < best[0]:
            best = (val, th)
    r = minimize(lambda th: fit(S, feats, (Bk @ np.array([np.cos(th[0]), np.sin(th[0])]))[:, None])[0],
                 [best[1]], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-15})
    u = Bk @ np.array([np.cos(r.x[0]), np.sin(r.x[0])])
    val, t, B = fit(S, feats, u[:, None])
    return val, u, t, B


def group_on(Y1, Y2, nseed=800):
    S1, S2 = np.exp(lsm(Y1)), np.exp(lsm(Y2))
    f1, f2 = Z1, c2[:, None]

    def tot(u):
        v = Bc @ (u / np.linalg.norm(u))
        return fit(S1, f1, v[:, None])[0] + fit(S2, f2, v[:, None])[0]
    U = rng.normal(size=(nseed, C - 1))
    vals = np.array([tot(u) for u in U])
    best = np.inf
    for i in np.argsort(vals)[:5]:
        r = minimize(tot, U[i], method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-15, "maxiter": 4000})
        best = min(best, r.fun)
    return best


def phi(t):
    return np.expm1(t) - t


def E_bound(Y, K, l_fit, rhat):
    """E_w of TQ3 part 2."""
    S = np.exp(lsm(Y))
    lam = logsumexp(Y[:, K], axis=1) - logsumexp(l_fit, axis=1)
    U = -Y[:, ~K] + lam[:, None]
    U = U - np.average(U, axis=0, weights=p)
    so = S[:, ~K] * p[:, None]
    tot = 0.0
    for i in range(U.shape[1]):
        f = lambda b: float(np.sum(so[:, i] * np.maximum(phi(U[:, i] - b + rhat), phi(U[:, i] - b - rhat))))
        bs = np.linspace(-30, 30, 6001)
        vals = [f(b) for b in bs]
        b0 = bs[int(np.argmin(vals))]
        r = minimize(lambda b: f(b[0]), [b0], method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-18})
        tot += r.fun
    return tot


def glued(Y1, Y2, own1, own2):
    """the glued build: V rows on K1 = u1, on K2 = u2; per-area read reproducing the own-block fit; t fitted."""
    (_, u1, t1, B1), (_, u2, t2, B2) = own1, own2
    c1 = Z1 @ B1.T            # (4, 1) own-fit amplitude on area 1 (line u1 is unit)
    c2v = c2[:, None] @ B2.T  # (4, 1)
    V = np.zeros((C, 1)); V[K1, 0] = u1; V[K2, 0] = u2
    loss = 0.0
    for Y, cc in [(Y1, c1), (Y2, c2v)]:
        S = np.exp(lsm(Y))
        L0 = cc @ V.T

        def obj(t):
            lq = lsm(L0 + t)
            return float(np.sum(p * np.sum(S * (np.log(S) - lq), axis=1))), (p[:, None] * (np.exp(lq) - S)).sum(0)
        r = minimize(obj, np.log(np.average(S, axis=0, weights=p)), jac=True, method="L-BFGS-B",
                     options={"gtol": 1e-12, "ftol": 1e-16})
        loss += r.fun
    l1 = t1 + c1 @ u1[None, :]
    l2 = t2 + c2v @ u2[None, :]
    rh1 = np.abs(c1[:, 0] - np.average(c1[:, 0], weights=p))
    rh2 = np.abs(c2v[:, 0] - np.average(c2v[:, 0], weights=p))
    return loss, l1, l2, rh1, rh2


out = []
for fam, (k1f, k2f) in [("even", (1, 1)), ("uneven", (1, 2))]:
    for m in [0.0, 1.0, 2.0, 4.0, 6.0, 8.0]:
        Y1, Y2 = logits(k1f * m, k2f * m)
        own1 = own_block_least(Y1, K1, Z1)
        own2 = own_block_least(Y2, K2, c2[:, None])
        Lc = own1[0] + own2[0]
        mu1 = float(np.sum(p * np.exp(lsm(Y1))[:, ~K1].sum(1)))
        mu2 = float(np.sum(p * np.exp(lsm(Y2))[:, ~K2].sum(1)))
        lower = max(0, 1 - mu1 / p.min()) * own1[0] + max(0, 1 - mu2 / p.min()) * own2[0]
        Lon = group_on(Y1, Y2)
        gl, l1, l2, rh1, rh2 = glued(Y1, Y2, own1, own2)
        E = E_bound(Y1, K1, l1, rh1) + E_bound(Y2, K2, l2, rh2)
        bound = Lc + E
        if lower > Lon + 1e-7:
            fails.append(f"{fam} m={m}: lower {lower} > Lon {Lon}")
        if gl > bound + 1e-9:
            fails.append(f"{fam} m={m}: glued {gl} > bound {bound}")
        if Lon > gl + 1e-7:
            fails.append(f"{fam} m={m}: search {Lon} worse than glued {gl}")
        row = dict(family=fam, m=m, mu=(mu1, mu2), Lc1=own1[0], Lc2=own2[0], lower=lower, Lon=Lon, glued=gl, bound=bound,
                   ratio=Lon / Lc if Lc > 0 else None)
        out.append(row)
        print(f"{fam:6s} m={m:3.1f} mu=({mu1:.4f},{mu2:.4f}) Lc1={own1[0]:.5f} Lc2={own2[0]:.1e} lower={lower:.5f} "
              f"Lon={Lon:.5f} glued={gl:.5f} bound={bound:.5f} Lon/Lc={Lon / Lc:.4f}")
print("fails:", len(fails), fails[:5])
json.dump(dict(rows=out, fails=fails), open(os.path.join(HERE, "ts_tq3.json"), "w"), indent=1, default=float)
