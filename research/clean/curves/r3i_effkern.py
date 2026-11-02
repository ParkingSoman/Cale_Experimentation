"""R3i (Statement 12.2): weights-only covariance from effective kernels in 3a space; formula from it."""
import numpy as np, torch, torch.nn.functional as F
from inception import *
import b3_train
from family import Data, IN4A, kl
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); keep = [i for i in range(480) if i not in fam]
w = lambda n: getattr(m, n).weight.detach().cpu()
E = torch.zeros(480, 256, 5, 5)
E[0:128, :, 2, 2] = w("mixed3b_1x1_pre_relu_conv")[:, :, 0, 0]
E[128:320, :, 1:4, 1:4] = torch.einsum("ocyx,ci->oiyx", w("mixed3b_3x3_pre_relu_conv"), w("mixed3b_3x3_bottleneck_pre_relu_conv")[:, :, 0, 0])
E[320:416] = torch.einsum("ocyx,ci->oiyx", w("mixed3b_5x5_pre_relu_conv"), w("mixed3b_5x5_bottleneck_pre_relu_conv")[:, :, 0, 0])
E[416:480, :, 1:4, 1:4] = (w("mixed3b_pool_reduce_pre_relu_conv")[:, :, 0, 0] / 9)[:, :, None, None]
E = E.reshape(480, -1).double(); G = E @ E.T
Sr = torch.from_numpy(np.load("cache/r3d_S3b.npy")); n = Sr[480, 480]; mr = Sr[:480, 480] / n; Cr = Sr[:480, :480] / n - torch.outer(mr, mr)
Ss = torch.from_numpy(np.load("cache/r3f_S3b_self.npy")); ms = Ss[:480, 480] / Ss[480, 480]
nz = lambda C: C / torch.sqrt(C.diagonal()[:, None] * C.diagonal()[None, :])
iu = torch.triu_indices(480, 480, 1)
print(f"pattern corr (normalised off-diagonal) G vs real cov: {np.corrcoef(nz(G)[iu[0], iu[1]].numpy(), nz(Cr)[iu[0], iu[1]].numpy())[0, 1]:.3f}")
fi = torch.tensor(fam); ki = torch.tensor(keep)
print(f"  family rows only (10 x 470): {np.corrcoef(nz(G)[fi][:, ki].flatten().numpy(), nz(Cr)[fi][:, ki].flatten().numpy())[0, 1]:.3f}")
def build(C, mu):
    Crr = C[keep][:, keep]; L = torch.linalg.solve(Crr + 1e-6 * torch.trace(Crr) / 470 * torch.eye(470, dtype=torch.float64), C[fam][:, keep].T).T
    c = (mu[fam] - L @ mu[keep]).float(); L = L.float()
    fm = b3_train.FamilyRA(m, fam, 1); fm.add_rb(m); fm.add_ra(m)
    with torch.no_grad():
        for cc in fm.cols: cc.zero_()
        for j, nme in enumerate(IN4A):
            wf = getattr(m, nme).weight.detach()[:, fam, 0, 0].cpu(); fm.dW[j].copy_((wf @ L) / fm.cscale[j]); fm.cbp[j].copy_((wf @ c) / 0.1)
    return fm.to(DEV)
def score(fm, idx, bs=100):
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]; lg = b3_train.forward_b3(m, fm, inputs_3b(m, to_3a(m, prep(D.X[b]))))
            out.append(kl(torch.from_numpy(D.T[b].astype(np.float32)).to(DEV), lg).cpu().numpy())
    v = np.concatenate(out); return round(float(v.mean()), 4), round(float(v.std() / np.sqrt(len(v))), 4)
Gs = G * (torch.trace(Cr) / torch.trace(G))          # one overall scale (Lambda is invariant to it anyway)
print("formula from G (weights only) + network-made means:", score(build(Gs, ms), D.dv))
