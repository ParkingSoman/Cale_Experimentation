# serves: TQ10
"""Check of Theorem TQ10 (COARSE-THEORY.md B.Q) on random small instances at the logit level of B.Q's setting.
Each instance: areas w with centred features ztil_w (points x, weights p), movement Y_w = B_w ztil_w, stacked rank > k.
Q0 = top-k eigenvectors of Sigma = sum p Y Y^T; K_w = Q0 A_w, T_w = (I - Q0 Q0^T) B_w; targets s^e = softmax(ybar_w +
(K_w + e T_w) ztil). Phi and Phi_free by weighted least squares (each point in H(s^0_x)). For e = 0.4, 0.2, 0.1, 0.05:
 - the TQ10 build (span Q0 + e Delta*, codes (A_w + e M*_w) ztil, free terms ybar_w + e beta*_w): its terms;
 - L^on(e) by local minimisation over always-on builds (V, Phi_w, t_w), restarts from random points and from the build.
Fails: build / (e^2 Phi) outside [0.95, 1.05] at e = 0.05; L^on / (e^2 Phi_free) below 0.95 at e = 0.05 (that would
refute part 3); Phi_free > Phi."""
import json
import os

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20261102)


def sqrtH(s):
    H = np.diag(s) - np.outer(s, s)
    ev, U = np.linalg.eigh(H); ev = np.clip(ev, 0, None)
    return (U * np.sqrt(ev)) @ U.T


def kl(s, l):
    ls = np.log(s); lq = l - logsumexp(l, axis=-1, keepdims=True)
    return np.sum(s * (ls - lq), axis=-1)


def instance():
    ny = int(rng.integers(4, 7)); nA = int(rng.integers(2, 4)); k = int(rng.integers(1, 3))
    areas = []
    for w in range(nA):
        d = int(rng.integers(2, 4)); n = int(rng.integers(8, 13))
        Z = rng.normal(size=(n, d)); Z -= Z.mean(0)
        B = rng.normal(size=(ny, d)) * rng.uniform(0.5, 1.5)
        B -= B.mean(0)                                     # image of P_perp
        ybar = rng.normal(size=ny); ybar -= ybar.mean()
        areas.append(dict(Z=Z, B=B, ybar=ybar, p=np.full(n, 1.0)))
    tot = sum(len(a["p"]) for a in areas)
    for a in areas:
        a["p"] = a["p"] / tot
    Sig = sum((a["p"][:, None] * (a["Z"] @ a["B"].T)).T @ (a["Z"] @ a["B"].T) for a in areas)
    ev, U = np.linalg.eigh(Sig); ev = ev[::-1]; U = U[:, ::-1]
    if not (ev[k - 1] > ev[k] + 1e-6 and ev[k] > 1e-6):
        return None
    Q0 = U[:, :k]
    for a in areas:
        a["A"] = Q0.T @ a["B"]; a["T"] = a["B"] - Q0 @ a["A"]
        a["s0"] = np.exp(a["ybar"] + a["Z"] @ (Q0 @ a["A"]).T - logsumexp(a["ybar"] + a["Z"] @ (Q0 @ a["A"]).T, axis=1, keepdims=True))
    return dict(ny=ny, k=k, Q0=Q0, areas=areas)


def phi(inst, free=False):
    """weighted least squares; unknowns Delta (ny x k), per area M_w (k x d) or m_x (k per point), beta_w (ny)."""
    ny, k, Q0 = inst["ny"], inst["k"], inst["Q0"]
    blocks = []; nD = ny * k
    nvar = nD
    offs = []
    for a in inst["areas"]:
        nm = (len(a["p"]) * k) if free else (k * a["Z"].shape[1])
        offs.append((nvar, nm)); nvar += nm + ny
    rows = []; rhs = []
    for wi, a in enumerate(inst["areas"]):
        o, nm = offs[wi]; d = a["Z"].shape[1]
        for xi in range(len(a["p"])):
            Lx = sqrtH(a["s0"][xi]) * np.sqrt(a["p"][xi])
            z = a["Z"][xi]; c0 = a["A"] @ z
            # r = Delta c0 + Q0 m + beta - T z ; columns: Delta (ny*k), m-part, beta
            G = np.zeros((ny, nvar))
            for i in range(ny):
                for j in range(k):
                    G[i, i * k + j] = c0[j]
            if free:
                G[:, o + xi * k: o + (xi + 1) * k] = Q0
            else:
                for j in range(k):
                    for t in range(d):
                        G[:, o + j * d + t] = Q0[:, j] * z[t]
            G[:, o + nm: o + nm + ny] = np.eye(ny)
            rows.append(Lx @ G); rhs.append(Lx @ (a["T"] @ z))
    Amat = np.vstack(rows); b = np.concatenate(rhs)
    sol = np.linalg.lstsq(Amat, b, rcond=None)[0]
    val = 0.5 * float(np.sum((Amat @ sol - b) ** 2))
    Delta = sol[:nD].reshape(ny, k)
    Ms = []; betas = []
    for wi, a in enumerate(inst["areas"]):
        o, nm = offs[wi]
        Ms.append(sol[o:o + nm]); betas.append(sol[o + nm:o + nm + ny])
    return val, Delta, Ms, betas


def targets(inst, e):
    out = []
    for a in inst["areas"]:
        y = a["ybar"] + a["Z"] @ (inst["Q0"] @ a["A"] + e * a["T"]).T
        out.append(np.exp(y - logsumexp(y, axis=1, keepdims=True)))
    return out


def build_terms(inst, e, Delta, Ms, betas):
    Q0, k = inst["Q0"], inst["k"]; S = targets(inst, e); tot = 0.0
    for wi, a in enumerate(inst["areas"]):
        d = a["Z"].shape[1]; M = Ms[wi].reshape(k, d)
        l = a["ybar"] + e * betas[wi] + a["Z"] @ ((Q0 + e * Delta) @ (a["A"] + e * M)).T
        tot += float(np.sum(a["p"] * kl(S[wi], l)))
    return tot


def lon(inst, e, Delta, Ms, betas, restarts=6):
    ny, k = inst["ny"], inst["k"]; S = targets(inst, e)
    dims = [a["Z"].shape[1] for a in inst["areas"]]
    def unpack(x):
        V = x[:ny * k].reshape(ny, k); o = ny * k; Ps = []; Ts = []
        for d in dims:
            Ps.append(x[o:o + k * d].reshape(k, d)); o += k * d
        for _ in dims:
            Ts.append(x[o:o + ny]); o += ny
        return V, Ps, Ts
    def f(x):
        V, Ps, Ts = unpack(x); tot = 0.0; g = np.zeros_like(x)
        gV = np.zeros((ny, k)); gP = []; gT = []
        for wi, a in enumerate(inst["areas"]):
            C = a["Z"] @ Ps[wi].T; l = Ts[wi] + C @ V.T
            lq = l - logsumexp(l, axis=1, keepdims=True); q = np.exp(lq)
            tot += float(np.sum(a["p"] * np.sum(S[wi] * (np.log(S[wi]) - lq), axis=1)))
            G = a["p"][:, None] * (q - S[wi])
            gV += G.T @ C; gP.append((G @ V).T @ a["Z"]); gT.append(G.sum(0))
        g[:ny * k] = gV.ravel(); o = ny * k
        for gp in gP:
            g[o:o + gp.size] = gp.ravel(); o += gp.size
        for gt in gT:
            g[o:o + ny] = gt; o += ny
        return tot, g
    starts = []
    x0 = [(inst["Q0"] + e * Delta).ravel()]
    for wi, a in enumerate(inst["areas"]):
        x0.append((a["A"] + e * Ms[wi].reshape(k, -1)).ravel())
    for wi, a in enumerate(inst["areas"]):
        x0.append(a["ybar"] + e * betas[wi])
    starts.append(np.concatenate(x0))
    nvar = starts[0].size
    for _ in range(restarts):
        starts.append(rng.normal(size=nvar))
    best = np.inf
    for x in starts:
        r = minimize(f, x, jac=True, method="L-BFGS-B", options=dict(maxiter=20000, gtol=1e-14, ftol=1e-16))
        best = min(best, r.fun)
    return best


fails = []; rows = []; n_ok = 0
while n_ok < 25:
    inst = instance()
    if inst is None:
        continue
    n_ok += 1
    P, Delta, Ms, betas = phi(inst); Pf = phi(inst, free=True)[0]
    if Pf > P * (1 + 1e-9) + 1e-14:
        fails.append(("free>phi", n_ok, Pf, P))
    row = dict(inst=n_ok, ny=inst["ny"], k=inst["k"], Phi=P, Phi_free=Pf, by_e=[])
    for e in [0.4, 0.2, 0.1, 0.05]:
        bt = build_terms(inst, e, Delta, Ms, betas)
        lo = lon(inst, e, Delta, Ms, betas)
        row["by_e"].append(dict(e=e, build_ratio=bt / (e * e * P), lon_ratio=lo / (e * e * P),
                                lon_over_free=lo / (e * e * Pf) if Pf > 0 else None))
    last = row["by_e"][-1]
    if not (0.95 <= last["build_ratio"] <= 1.05):
        fails.append(("build", n_ok, last["build_ratio"]))
    if last["lon_over_free"] is not None and last["lon_over_free"] < 0.95:
        fails.append(("lower", n_ok, last["lon_over_free"]))
    rows.append(row)
    print(n_ok, "ny", inst["ny"], "k", inst["k"], " build/e2Phi:",
          ["%.3f" % r["build_ratio"] for r in row["by_e"]], " Lon/e2Phi:", ["%.3f" % r["lon_ratio"] for r in row["by_e"]],
          " Phi_free/Phi %.3f" % (Pf / P), flush=True)
print("instances", n_ok, "fails", len(fails), fails[:5])
json.dump(dict(fails=fails, rows=rows), open(os.path.join(HERE, "tq_small.json"), "w"), indent=1)
