"""X3: an imprint-neutral, self-labeled measure q (channels C1 + C2 at 3b unchanged) that changes the optimal write."""
import numpy as np, torch
from scipy.optimize import nnls
from inception import *
from family import Data
from w_fisher import load_build, real_kl, intact, coarse, A_cols, fisher_grads
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); M = load_build(m, fam, "L-k3-m-lr3e-2"); A = A_cols(m, fam)
for p in m.parameters(): p.requires_grad_(False)
pool = np.concatenate([np.load("cache/e1_self_256.npy"), np.load("cache/e7_bnd_1024_128_0.npy"), np.load("cache/c31_fv.npy")[:104]]); ds = np.load("cache/n8_ds.npy")
import sys
with torch.no_grad():
    lg = torch.cat([__import__('w_fisher').upper(m, intact(m, fam, inputs_3b(m, to_3a(m, prep(pool[i:i + 32]))))[0]) for i in range(0, len(pool), 32)])
    u = (1 - (F.softmax(lg, 1) ** 2).sum(1)).cpu().numpy()
keep = u > .1; print(f"unsure inputs in the pool (1 - sum p^2 > .1): {keep.sum()} (inv {keep[:256].sum()}, bnd {keep[256:1280].sum()}, fv {keep[1280:].sum()})", flush=True)
ds = ds * keep[:, None]
sc = np.median(np.linalg.norm(ds[keep], axis=1)); X = ds / sc; BIG = 100.0
Xk = X[keep]; qk, r = nnls(np.vstack([Xk.T, BIG * np.ones((1, len(Xk)))]), np.r_[np.zeros(480), BIG], maxiter=50000); q = np.zeros(len(X)); q[keep] = qk; q = q / q.sum()
res = np.linalg.norm(q @ X); print(f"imprint-neutral q: residual |sum q d| / median |d| = {res:.4f}; support {np.sum(q > 0)} inputs; by source inv {q[:256].sum():.3f} bnd {q[256:1280].sum():.3f} fv {q[1280:].sum():.3f}", flush=True)
sizes = [a.shape[0] for a in A]; P = sum(sizes) * 4
def wsolve(X_img, w):
    XtX = torch.zeros(P, P, dtype=torch.float64); Xty = torch.zeros(P, dtype=torch.float64); sup = np.where(w > 0)[0]
    for i in range(0, len(sup), 25):
        b = sup[i:i + 25]; sw = torch.tensor(np.sqrt(w[b]), dtype=torch.float32, device=DEV)
        with torch.no_grad(): ins = inputs_3b(m, to_3a(m, prep(X_img[b]))); outs, z1, z2 = intact(m, fam, ins); c1, c2 = coarse(M, ins)
        G, _ = fisher_grads(m, outs, 10)
        with torch.no_grad():
            for g in G:
                feats, y = [], 0.0
                for j in range(4):
                    c = c1 if j < 3 else c2; z = z1 if j < 3 else z2; gj = g[j] * sw[:, None, None, None]
                    feats.append(torch.einsum("bohw,bkhw->bok", gj, torch.cat([c, torch.ones_like(c[:, :1])], 1)).reshape(len(b), -1))
                    y = y + torch.einsum("bohw,oi,bihw->b", gj, A[j], z)
                Xf = torch.cat(feats, 1).cpu().double(); XtX += Xf.T @ Xf; Xty += Xf.T @ y.cpu().double()
    return XtX, Xty
Nq, mq = wsolve(pool, q)
ld = lambda t: torch.load(f"cache/bv_{t}.pt"); N1 = (ld("R1")["XtX"] + ld("R2")["XtX"]) / (ld("R1")["n"] + ld("R2")["n"]); m1 = (ld("R1")["Xty"] + ld("R2")["Xty"]) / (ld("R1")["n"] + ld("R2")["n"])
N2, m2 = (N1 + Nq) / 2, (m1 + mq) / 2
reg = lambda N: N + 1e-6 * torch.trace(N) / P * torch.eye(P, dtype=N.dtype)
w1, w2 = torch.linalg.solve(reg(N1), m1), torch.linalg.solve(reg(N2), m2); d = w1 - w2
bound = 0.5 * float(d @ torch.linalg.solve(torch.linalg.inv(reg(N1)) + torch.linalg.inv(reg(N2)), d))
print(f"bound (second-order units): {bound:.4f}; for scale, w1's own loss scale (w1^T N1 w1) {float(w1 @ N1 @ w1):.4f}", flush=True)
def unpack(w):
    cols, cbs, o = [], [], 0
    for s in sizes:
        Wt = w[o:o + 4 * s].reshape(s, 4).float(); o += 4 * s; cols.append(Wt[:, :3].to(DEV)); cbs.append(Wt[:, 3].to(DEV))
    return cols, cbs
B = {"w*(p1): real unsure": (M, *unpack(w1)), "w*(p2): real unsure + imprint-neutral self-labeled q": (M, *unpack(w2))}
K = real_kl(m, fam, D, D.dv, B)
for k, v in K.items(): print(f"  {k:52s} dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})")
dd = list(K.values())[1] - list(K.values())[0]; print(f"  paired w*(p2) - w*(p1) on real images: {dd.mean():+.4f} (se {dd.std() / np.sqrt(len(dd)):.4f})")
np.save("cache/x3_q_unsure.npy", q); print(f"q mean 1 - sum p^2 {float((q * u).sum()):.3f}")
