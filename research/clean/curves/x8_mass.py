"""X8: imprint-neutral (relative tolerance) self-labeled mixture WITH a write-mass floor; effect at large eps."""
import numpy as np, torch
from scipy.optimize import linprog
import x3_counter as X
from w_fisher import real_kl
keep = np.where(X.keep)[0]; tr1 = float(torch.trace(X.N1))
tau = np.array([float(torch.trace(X.wsolve(X.pool[i:i + 1], np.ones(1))[0])) / tr1 for i in keep]); print(f"per-input write mass tau: median {np.median(tau):.3f}, share with tau >= 1: {np.mean(tau >= 1):.2f}", flush=True)
import x4_adv as X4   # gives obj (first-order write shift per input) on the same unsure pool
Dk = X.X[keep]; mag = np.abs(Dk).mean(1); rel = 0.02
A_ub = np.vstack([Dk.T - rel * mag[None, :], -Dk.T - rel * mag[None, :], -tau[None, :]]); b_ub = np.r_[np.zeros(960), -1.0]
lp = linprog(-X4.obj, A_ub=A_ub, b_ub=b_ub, A_eq=np.ones((1, len(keep))), b_eq=[1.0], bounds=(0, None), method="highs")
print(f"LP status {lp.status} ({lp.message})", flush=True)
q = np.zeros(len(X.pool)); q[keep] = lp.x
print(f"q: support {np.sum(q > 1e-9)}; write mass {float(lp.x @ tau):.2f}; imprint residual / own mean |d|: {np.linalg.norm(q @ X.X) / float(lp.x @ mag):.3f}; mean 1 - sum p^2 {float((q * X.u).sum()):.3f}", flush=True)
Nq, mq = X.wsolve(X.pool, q); B = {"w*(p1)": (X.M, *X.unpack(X.w1)), "w*(q) alone": (X.M, *X.unpack(torch.linalg.solve(X.reg(Nq), mq)))}
for e in [1.0, 10.0, 100.0, 1000.0]:
    w2 = torch.linalg.solve(X.reg((X.N1 + e * Nq) / (1 + e)), (X.m1 + e * mq) / (1 + e)); B[f"w*(p1 + {e:g} q)"] = (X.M, *X.unpack(w2))
K = real_kl(X.m, X.fam, X.D, X.D.dv, B); base = K["w*(p1)"]
for k, v in K.items(): dd = v - base; print(f"  {k:20s} dev-5000 KL {v.mean():.4f} | paired vs w*(p1) {dd.mean():+.4f} (se {dd.std() / np.sqrt(len(dd)):.4f})")
np.save("cache/x8_q.npy", q)
