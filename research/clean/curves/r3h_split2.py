"""R3h (Statement 11.5'): split the network-made gap into covariance (pattern) and means, from saved moments."""
import numpy as np, torch
from inception import *
import b3_train
from family import Data, IN4A, kl
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); keep = [i for i in range(480) if i not in fam]
def cov_mu(S):
    S = torch.from_numpy(S); n = S[480, 480]; mu = S[:480, 480] / n; return S[:480, :480] / n - torch.outer(mu, mu), mu
Cr, mr = cov_mu(np.load("cache/r3d_S3b.npy")); Cs, ms = cov_mu(np.load("cache/r3f_S3b_self.npy"))
def lam(C, mu):
    Crr = C[keep][:, keep]; L = torch.linalg.solve(Crr + 1e-6 * torch.trace(Crr) / 470 * torch.eye(470, dtype=torch.float64), C[fam][:, keep].T).T
    return L.float(), (mu[fam] - L @ mu[keep]).float()
def build(L, c):
    fm = b3_train.FamilyRA(m, fam, 1); fm.add_rb(m); fm.add_ra(m)
    with torch.no_grad():
        for cc in fm.cols: cc.zero_()
        for j, n in enumerate(IN4A):
            w = getattr(m, n).weight.detach()[:, fam, 0, 0].cpu(); fm.dW[j].copy_((w @ L) / fm.cscale[j]); fm.cbp[j].copy_((w @ c) / 0.1)
    return fm.to(DEV)
def score(fm, idx, bs=100):
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]; lg = b3_train.forward_b3(m, fm, inputs_3b(m, to_3a(m, prep(D.X[b]))))
            out.append(kl(torch.from_numpy(D.T[b].astype(np.float32)).to(DEV), lg).cpu().numpy())
    v = np.concatenate(out); return v.mean(), v.std() / np.sqrt(len(v))
Lrr, _ = lam(Cr, mr); Lss, _ = lam(Cs, ms)
for name, L, mu in [("real cov + real means", Lrr, mr), ("real cov + made means", Lrr, ms), ("made cov + real means", Lss, mr), ("made cov + made means", Lss, ms)]:
    c = (mu[fam] - L.double() @ mu[keep]).float(); print(f"  {name:22s} {score(build(L, c), D.dv)}", flush=True)
print("mean levels: median made/real", float(np.median((ms / (mr + 1e-9)).numpy())), " Spearman", __import__('scipy.stats').stats.spearmanr(mr.numpy(), ms.numpy())[0])
