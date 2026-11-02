"""G2 (Statement 12.4', premise): Gaussian SCALE MIXTURE model of the 3b pre-activations vs the true post-ReLU covariance.
v = |p - mu|^2 / E|p - mu|^2 per position; Cn = E[(p - mu)(p - mu)^T / v]. FITTED, same 2000 train images as G1."""
import numpy as np, torch
from inception import *
from family import Data
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); keep = [i for i in range(480) if i not in fam]
tr = np.sort(np.random.default_rng(5).choice(D.tr, 2000, replace=False))
G1 = np.load("cache/g1_gauss.npz"); mu = torch.from_numpy(G1["mup"]); Cp = torch.from_numpy(G1["Cp"]); Cr = torch.from_numpy(G1["Cr"])
tot = torch.trace(Cp)
Cn = torch.zeros(480, 480, dtype=torch.float64); vs = []; n = 0
with torch.no_grad():
    for i in range(0, len(tr), 50):
        p = pre_3b(m, inputs_3b(m, to_3a(m, prep(D.X[tr[i:i + 50]])))[0]).permute(0, 2, 3, 1).reshape(-1, 480).cpu().double() - mu
        v = (p * p).sum(1) / tot; v = v.clamp_min(1e-6)
        Cn += (p / v.sqrt()[:, None]).T @ (p / v.sqrt()[:, None]); n += len(p); vs.append(v[::7].clone())
Cn /= n; vs = torch.cat(vs)
print(f"v law: mean {vs.mean():.3f}, median {vs.median():.3f}, 90th pct {vs.quantile(.9):.3f}, 99th pct {vs.quantile(.99):.3f}, kurtosis-like E[v^2]/E[v]^2 {(vs ** 2).mean() / vs.mean() ** 2:.2f}")
g = torch.Generator().manual_seed(0); L = torch.linalg.cholesky(Cn + 1e-8 * torch.eye(480, dtype=torch.float64))
S = torch.zeros(481, 481, dtype=torch.float64)
for _ in range(20):
    vv = vs[torch.randint(len(vs), (20000,), generator=g)]
    x = (mu + vv.sqrt()[:, None] * (torch.randn(20000, 480, generator=g, dtype=torch.float64) @ L.T)).clamp_min(0)
    H = torch.cat([x, torch.ones(len(x), 1, dtype=torch.float64)], 1); S += H.T @ H
N = S[480, 480]; mug = S[:480, 480] / N; Cg = S[:480, :480] / N - torch.outer(mug, mug)
r = (Cg.diagonal() / Cr.diagonal()).numpy()
print(f"(a) var ratio model/truth: median {np.median(r):.2f}, within x1.5 {np.mean((r > 1 / 1.5) & (r < 1.5)):.2f}, within x2 {np.mean((r > .5) & (r < 2)):.2f}; mean ratio {np.median((mug / torch.from_numpy(G1['mur'])).numpy()):.2f}")
nz = lambda C: C / torch.sqrt(C.diagonal()[:, None] * C.diagonal()[None, :])
iu = torch.triu_indices(480, 480, 1); fi, ki = torch.tensor(fam), torch.tensor(keep)
a = np.corrcoef(nz(Cg)[iu[0], iu[1]].numpy(), nz(Cr)[iu[0], iu[1]].numpy())[0, 1]
f = np.corrcoef(nz(Cg)[fi][:, ki].flatten().numpy(), nz(Cr)[fi][:, ki].flatten().numpy())[0, 1]
def lam(C):
    Crr = C[keep][:, keep]; return torch.linalg.solve(Crr + 1e-6 * torch.trace(Crr) / 470 * torch.eye(470, dtype=torch.float64), C[fam][:, keep].T).T
cs = torch.nn.functional.cosine_similarity(lam(Cg), lam(Cr), 1).numpy()
print(f"(b) pattern vs true post-ReLU: all pairs {a:.3f}, family rows {f:.3f}")
print(f"(c) Lambda-row cos: {np.round(cs, 2)} median {np.median(cs):.3f}")
np.savez("cache/g2_gsm.npz", Cn=Cn.numpy(), vs=vs.numpy(), Cg=Cg.numpy(), mug=mug.numpy())
