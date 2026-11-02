"""G1 (NEURON-MATH Statement 12.4, premise): Gaussian-model post-ReLU covariance from the TRUE pre-activation moments vs
the true post-ReLU covariance. FITTED moments, 2000 train images (same as R3d). No score."""
import numpy as np, torch
from inception import *
from family import Data, pool3
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); keep = [i for i in range(480) if i not in fam]
tr = np.sort(np.random.default_rng(5).choice(D.tr, 2000, replace=False))
z = lambda: torch.zeros(481, 481, dtype=torch.float64)
Sp, Sr, Sq = z(), z(), z()                      # pre, relu(pre), pool3(relu(pre)); each with a constant column
def acc(S, x):
    H = torch.cat([x.permute(0, 2, 3, 1).reshape(-1, 480), torch.ones(x.shape[0] * x.shape[2] * x.shape[3], 1, device=DEV)], 1).cpu().double()
    S += H.T @ H
with torch.no_grad():
    for i in range(0, len(tr), 50):
        p = pre_3b(m, inputs_3b(m, to_3a(m, prep(D.X[tr[i:i + 50]])))[0])
        acc(Sp, p); acc(Sr, F.relu(p)); acc(Sq, pool3(F.relu(p)))
def cov(S):
    n = S[480, 480]; mu = S[:480, 480] / n; return S[:480, :480] / n - torch.outer(mu, mu), mu
Cp, mup = cov(Sp); Cr, mur = cov(Sr); Cq, _ = cov(Sq)
g = torch.Generator().manual_seed(0); Lc = torch.linalg.cholesky(Cp + 1e-8 * torch.eye(480, dtype=torch.float64))
Sg = z()
for _ in range(20):
    x = (mup + torch.randn(20000, 480, generator=g, dtype=torch.float64) @ Lc.T).clamp_min(0)
    H = torch.cat([x, torch.ones(len(x), 1, dtype=torch.float64)], 1); Sg += H.T @ H
Cg, mug = cov(Sg)
r = (Cg.diagonal() / Cr.diagonal()).numpy(); zsc = (mup / Cp.diagonal().sqrt()).numpy()
print(f"(a) var ratio model/truth: median {np.median(r):.2f}, within x1.5 {np.mean((r > 1 / 1.5) & (r < 1.5)):.2f}, "
      f"within x2 {np.mean((r > .5) & (r < 2)):.2f}")
for lo, hi in [(-9, -1.5), (-1.5, -.5), (-.5, .5), (.5, 9)]:
    s = (zsc >= lo) & (zsc < hi)
    if s.any(): print(f"    pre mean/sd in [{lo},{hi}): n {s.sum():3d}, median ratio {np.median(r[s]):.2f}")
print(f"    mean ratio model/truth: median {np.median((mug / mur).numpy()):.2f}")
nz = lambda C: C / torch.sqrt(C.diagonal()[:, None] * C.diagonal()[None, :])
iu = torch.triu_indices(480, 480, 1); fi, ki = torch.tensor(fam), torch.tensor(keep)
def pc(A, B): return np.corrcoef(nz(A)[iu[0], iu[1]].numpy(), nz(B)[iu[0], iu[1]].numpy())[0, 1], \
    np.corrcoef(nz(A)[fi][:, ki].flatten().numpy(), nz(B)[fi][:, ki].flatten().numpy())[0, 1]
def lam(C):
    Crr = C[keep][:, keep]; return torch.linalg.solve(Crr + 1e-6 * torch.trace(Crr) / 470 * torch.eye(470, dtype=torch.float64), C[fam][:, keep].T).T
def lc(A, B):
    a, b = lam(A), lam(B); return np.round(torch.nn.functional.cosine_similarity(a, b, 1).numpy(), 2)
for name, A in [("Gaussian model", Cg), ("no-ReLU baseline", Cp)]:
    a, f = pc(A, Cr); print(f"(b) {name:17s} vs true post-ReLU: all pairs {a:.3f}, family rows {f:.3f}")
    print(f"(c)    Lambda-row cos vs true post-ReLU: {lc(A, Cr)}")
a, f = pc(Cr, Cq); print(f"diag: true post-ReLU unpooled vs pooled: all pairs {a:.3f}, family rows {f:.3f}; Lambda cos {lc(Cr, Cq)}")
a, f = pc(Cg, Cq); print(f"diag: Gaussian model vs true pooled: all pairs {a:.3f}, family rows {f:.3f}; Lambda cos {lc(Cg, Cq)}")
np.savez("cache/g1_gauss.npz", Cp=Cp.numpy(), mup=mup.numpy(), Cr=Cr.numpy(), mur=mur.numpy(), Cq=Cq.numpy(), Cg=Cg.numpy(), mug=mug.numpy())
