"""X9: direction-only imprint consistency (combined imbalance parallel to D). Max write mass vs tolerance; then a counterexample."""
import numpy as np, torch
from scipy.optimize import linprog
import x3_counter as X
from w_fisher import real_kl
keep = np.where(X.keep)[0]; tr1 = float(torch.trace(X.N1))
tau = np.array([float(torch.trace(X.wsolve(X.pool[i:i + 1], np.ones(1))[0])) / tr1 for i in keep]); np.save("cache/x9_tau.npy", tau)
Dw = np.load("cache/n6_sgd.npz")["D"]; t = Dw / np.linalg.norm(Dw); Dk = X.X[keep]; Pk = Dk - np.outer(Dk @ t, t); mag = np.abs(Dk).mean(1)
res = {}
for rel in [.02, .05, .1, .2]:
    A_ub = np.vstack([Pk.T - rel * mag[None, :], -Pk.T - rel * mag[None, :]])
    lp = linprog(-tau, A_ub=A_ub, b_ub=np.zeros(960), A_eq=np.ones((1, len(keep))), b_eq=[1.0], bounds=(0, None), method="highs")
    res[rel] = lp.x if lp.status == 0 else None
    print(f"rel {rel}: max write mass {(-lp.fun if lp.status == 0 else float('nan')):.3f} (status {lp.status}); support {np.sum(lp.x > 1e-9) if lp.status == 0 else 0}", flush=True)
import x4_adv as X4
for rel in [.02, .05, .1, .2]:
    A_ub = np.vstack([Pk.T - rel * mag[None, :], -Pk.T - rel * mag[None, :], -tau[None, :]])
    lp = linprog(-X4.obj, A_ub=A_ub, b_ub=np.r_[np.zeros(960), -1.0], A_eq=np.ones((1, len(keep))), b_eq=[1.0], bounds=(0, None), method="highs")
    if lp.status != 0: print(f"rel {rel}: no direction-consistent mixture with write mass >= 1", flush=True); continue
    q = np.zeros(len(X.pool)); q[keep] = lp.x; Nq, mq = X.wsolve(X.pool, q); B = {"w*(p1)": (X.M, *X.unpack(X.w1))}
    print(f"rel {rel}: counterexample q: support {np.sum(q > 1e-9)}, write mass {float(lp.x @ tau):.2f}, cos(sum q d, D) {float((q @ X.X) @ t / np.linalg.norm(q @ X.X)):.3f}", flush=True)
    for e in [10.0, 100.0, 1000.0]:
        w2 = torch.linalg.solve(X.reg((X.N1 + e * Nq) / (1 + e)), (X.m1 + e * mq) / (1 + e)); B[f"eps {e:g}"] = (X.M, *X.unpack(w2))
    K = real_kl(X.m, X.fam, X.D, X.D.dv, B); base = K["w*(p1)"]
    for k, v in K.items(): dd = v - base; print(f"    {k:8s} dev-5000 KL {v.mean():.4f} | paired vs w*(p1) {dd.mean():+.4f} (se {dd.std() / np.sqrt(len(dd)):.4f})")
    np.save(f"cache/x9_q_{rel}.npy", q); break
