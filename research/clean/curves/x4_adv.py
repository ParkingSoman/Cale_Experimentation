"""X4: adversarial consistent mixture. Imprint-neutral, self-labeled, unsure q that moves the optimal write most."""
import numpy as np, torch
from scipy.optimize import linprog
import x3_counter as X   # reuses pool, unsure mask, energy vectors, w1, N1, m1, wsolve, unpack (re-runs X3 quickly)
from inception import *
from w_fisher import real_kl, intact, coarse, fisher_grads
m, M, A, D, fam, P = X.m, X.M, X.A, X.D, X.fam, X.P
keep = np.where(X.keep)[0]; R = []
for i in range(0, len(keep), 25):                      # per-input r_s = m_s - N_s w1 (1472-dim)
    b = keep[i:i + 25]
    with torch.no_grad(): ins = inputs_3b(m, to_3a(m, prep(X.pool[b]))); outs, z1, z2 = intact(m, fam, ins); c1, c2 = coarse(M, ins)
    G, _ = fisher_grads(m, outs, 10); rs = torch.zeros(len(b), P, dtype=torch.float64)
    with torch.no_grad():
        for g in G:
            feats, y = [], 0.0
            for j in range(4):
                c = c1 if j < 3 else c2; z = z1 if j < 3 else z2
                feats.append(torch.einsum("bohw,bkhw->bok", g[j], torch.cat([c, torch.ones_like(c[:, :1])], 1)).reshape(len(b), -1)); y = y + torch.einsum("bohw,oi,bihw->b", g[j], A[j], z)
            Xf = torch.cat(feats, 1).cpu().double(); rs += Xf * (y.cpu().double() - Xf @ X.w1)[:, None]
    R.append(rs)
R = torch.cat(R).numpy(); N1i = torch.linalg.inv(X.reg(X.N1)).numpy(); v = N1i @ R.mean(0); obj = R @ v
Dk = X.X[keep]
tol = 0.002
lp = linprog(-obj, A_ub=np.vstack([Dk.T, -Dk.T]), b_ub=np.full(960, tol), A_eq=np.ones((1, len(keep))), b_eq=[1.0], bounds=(0, None), method="highs")
print(f"LP status {lp.status} ({lp.message}); support {np.sum(lp.x > 1e-9)}", flush=True)
q = np.zeros(len(X.pool)); q[keep] = lp.x; print(f"imprint residual |sum q d| = {np.linalg.norm(q @ X.X):.2e}; q mean 1 - sum p^2 {float((q * X.u).sum()):.3f}")
Nq, mq = X.wsolve(X.pool, q); B = {"w*(p1): real unsure": (M, *X.unpack(X.w1))}
for eps in [1.0, 10.0]:
    N2, m2 = (X.N1 + eps * Nq) / (1 + eps), (X.m1 + eps * mq) / (1 + eps); w2 = torch.linalg.solve(X.reg(N2), m2); d = X.w1 - w2
    bound = 0.5 * float(d @ torch.linalg.solve(torch.linalg.inv(X.reg(X.N1)) + torch.linalg.inv(X.reg(N2)), d))
    print(f"eps {eps}: bound (second-order units) {bound:.4f}", flush=True); B[f"w*(p2), eps = {eps}"] = (M, *X.unpack(w2))
K = real_kl(m, fam, D, D.dv, B); base = K["w*(p1): real unsure"]
for k, v_ in K.items(): d_ = v_ - base; print(f"  {k:24s} dev-5000 KL {v_.mean():.4f} (se {v_.std() / np.sqrt(len(v_)):.4f}) | paired vs w*(p1) {d_.mean():+.4f} (se {d_.std() / np.sqrt(len(d_)):.4f})")
np.save("cache/x4_q.npy", q)
