# serves: TQ7
"""Check of Proposition TQ7 (COARSE-THEORY.md B.Q), in coordinates of the image of P_perp (dimension d = n_y - 1).
Random instances: areas with covariances Sigma_w (PSD, d x d), a metric Hb (PD), area metrics
H_w = Hb^1/2 (I + E_w) Hb^1/2 with E_w symmetric, |E_w|_op <= eta (so (1 - eta) Hb <= H_w <= (1 + eta) Hb).
psi_w(Q) = tr(H_w Sigma_w) - tr((Q^T H_w Q)^-1 Q^T H_w Sigma_w H_w Q). Checks (each one-sided, so a local optimiser
can only make them harder to pass):
 (1 lower) every Q found by direct minimisation has (1/2) sum psi_w(Q) >= (1 - eta) tail_k(M_b) / 2;
 (2) (1/2) sum psi_w(Q_b) <= ((1 + eta)/(1 - eta)) x (the least found), and <= (1 + eta) tail_k(M_b) / 2 (1 upper);
 (3) eta = 0: Q_b attains the least found (TI4 part 2)."""
import json
import os

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1101)
fails = []; rows = []


def sqrtm_pd(A):
    ev, U = np.linalg.eigh(A)
    return (U * np.sqrt(ev)) @ U.T, (U / np.sqrt(ev)) @ U.T


def psi_sum(Q, Hs, Ss):
    tot = 0.0
    for H, S in zip(Hs, Ss):
        G = Q.T @ H @ Q
        tot += np.trace(H @ S) - np.trace(np.linalg.solve(G, Q.T @ H @ S @ H @ Q))
    return tot


for trial in range(120):
    d = int(rng.integers(3, 8)); k = int(rng.integers(1, d)); nw = int(rng.integers(1, 5))
    eta = float(rng.choice([0.0, 0.05, 0.2, 0.5, 0.8]))
    B = rng.normal(size=(d, d)); Hb = B @ B.T + 0.2 * np.eye(d)
    Hh, Hih = sqrtm_pd(Hb)
    Hs, Ss = [], []
    for _ in range(nw):
        E = rng.normal(size=(d, d)); E = (E + E.T) / 2
        E *= eta / max(np.abs(np.linalg.eigvalsh(E)).max(), 1e-12) * rng.uniform(0, 1)
        Hs.append(Hh @ (np.eye(d) + E) @ Hh)
        C = rng.normal(size=(d, int(rng.integers(1, d + 1)))); Ss.append(C @ C.T)
    Mb = Hh @ sum(Ss) @ Hh
    ev, U = np.linalg.eigh(Mb)
    tail = float(ev[: d - k].sum())
    Qb = np.linalg.qr(Hih @ U[:, ::-1][:, :k])[0]
    vb = 0.5 * psi_sum(Qb, Hs, Ss)
    best = np.inf
    for _ in range(15):
        f = lambda x: 0.5 * psi_sum(np.linalg.qr(x.reshape(d, k))[0], Hs, Ss)
        r = minimize(f, rng.normal(size=d * k), method="BFGS", options=dict(gtol=1e-12, maxiter=5000))
        best = min(best, r.fun)
    best = min(best, vb)
    tol = 1e-8 * max(1.0, tail)
    if best < (1 - eta) * tail / 2 - tol:
        fails.append(("lower", trial, best, (1 - eta) * tail / 2))
    if vb > (1 + eta) * tail / 2 + tol:
        fails.append(("upper", trial, vb, (1 + eta) * tail / 2))
    if eta < 1 and vb > (1 + eta) / (1 - eta) * best + tol:
        fails.append(("ratio", trial, vb, best))
    if eta == 0 and vb > best + 1e-7 * max(1, best):
        fails.append(("eta0", trial, vb, best))
    rows.append(dict(trial=trial, d=d, k=k, areas=nw, eta=eta, shared=vb, least_found=best, tail_half=tail / 2,
                     ratio=vb / best if best > 0 else 1.0))
print("checks", len(rows), "fails", len(fails), fails[:5])
for e in [0.0, 0.05, 0.2, 0.5, 0.8]:
    rr = [r["ratio"] for r in rows if r["eta"] == e]
    if rr:
        print("eta %.2f: worst shared / least found = %.4f (bound %.4f), n = %d"
              % (e, max(rr), (1 + e) / (1 - e), len(rr)))
json.dump(dict(fails=fails, rows=rows), open(os.path.join(HERE, "tq_shared.json"), "w"), indent=1)
