"""R3g (Statement 11.4): hybrid Lambda builds, coefficients from one moment set and constant from the other."""
import numpy as np, torch
from inception import *
import b3_train
from family import Data, IN4A, kl
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); keep = [i for i in range(480) if i not in fam]
def lam(S):
    S = torch.from_numpy(S); S = S / S[480, 480]; R = keep + [480]; Srr = S[R][:, R]
    return torch.linalg.solve(Srr + 1e-6 * torch.trace(Srr) / len(R) * torch.eye(len(R), dtype=torch.float64), S[fam][:, R].T).T.float()
Lr, Ls = lam(np.load("cache/r3d_S3b.npy")), lam(np.load("cache/r3f_S3b_self.npy"))
def build(coef, const):
    fm = b3_train.FamilyRA(m, fam, 1); fm.add_rb(m); fm.add_ra(m)
    with torch.no_grad():
        for c in fm.cols: c.zero_()
        for j, n in enumerate(IN4A):
            w = getattr(m, n).weight.detach()[:, fam, 0, 0].cpu()
            fm.dW[j].copy_((w @ coef) / fm.cscale[j]); fm.cbp[j].copy_((w @ const) / 0.1)
    return fm.to(DEV)
def perimage(fm, idx, bs=100):
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]; lg = b3_train.forward_b3(m, fm, inputs_3b(m, to_3a(m, prep(D.X[b]))))
            out.append(kl(torch.from_numpy(D.T[b].astype(np.float32)).to(DEV), lg).cpu().numpy())
    return np.concatenate(out)
K = {"real coef + real const": perimage(build(Lr[:, :470], Lr[:, 470]), D.dv),
     "real coef + made const": perimage(build(Lr[:, :470], Ls[:, 470]), D.dv),
     "made coef + real const": perimage(build(Ls[:, :470], Lr[:, 470]), D.dv),
     "made coef + made const": perimage(build(Ls[:, :470], Ls[:, 470]), D.dv)}
for k, v in K.items(): print(f"  {k:24s} {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})")
