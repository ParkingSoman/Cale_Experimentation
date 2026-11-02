"""R3e: Prop. 11.1 closed-form re-routing (k = 0) from second moments of (a) 2000 real train images (FITTED) and (b) the 256
network-made class inversions (no real images), paired per-image against the trained B3 rulers. dev-5000."""
import numpy as np, torch
from inception import *
import b3_train
from family import Data, IN4A, pool3, kl
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); keep = [i for i in range(480) if i not in fam]
def moments(X, flip=True):
    S = torch.zeros(481, 481, dtype=torch.float64)
    with torch.no_grad():
        for i in range(0, len(X), 32):
            xb = np.asarray(X[i:i + 32]); xs = [xb, xb[:, :, ::-1]] if flip else [xb]
            for xx in xs:
                x = pool3(F.relu(pre_3b(m, inputs_3b(m, to_3a(m, prep(np.ascontiguousarray(xx))))[0])))
                H = torch.cat([x.permute(0, 2, 3, 1).reshape(-1, 480), torch.ones(x.shape[0] * 196, 1, device=DEV)], 1).cpu().double()
                S += H.T @ H
    return S
def build(S):
    R = keep + [480]; Srr = S[R][:, R]; Sfr = S[fam][:, R]
    Lam = torch.linalg.solve(Srr + 1e-6 * torch.trace(Srr) / len(R) * torch.eye(len(R), dtype=torch.float64), Sfr.T).T.float()
    fm = b3_train.FamilyRA(m, fam, 1); fm.add_rb(m); fm.add_ra(m)
    with torch.no_grad():
        for c in fm.cols: c.zero_()
        for j, n in enumerate(IN4A):
            w = getattr(m, n).weight.detach()[:, fam, 0, 0].cpu()
            fm.dW[j].copy_((w @ Lam[:, :470]) / fm.cscale[j]); fm.cbp[j].copy_((w @ Lam[:, 470]) / 0.1)
    return fm.to(DEV)
def perimage(fm, idx, bs=100):
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]; lg = b3_train.forward_b3(m, fm, inputs_3b(m, to_3a(m, prep(D.X[b]))))
            out.append(kl(torch.from_numpy(D.T[b].astype(np.float32)).to(DEV), lg).cpu().numpy())
    return np.concatenate(out)
S_real = torch.from_numpy(np.load("cache/r3d_S3b.npy"))           # 2000 real train images (no flips), from r3d
S_self = moments(np.load("cache/e1_self_256.npy", mmap_mode="r"))  # 256 network-made inputs + mirror flips
K = {"formula, real moments (FITTED)": perimage(build(S_real), D.dv),
     "formula, network-made moments": perimage(build(S_self), D.dv),
     "trained B3 k=0": perimage(b3_train.load_b3(m, "B3-k0"), D.dv),
     "trained B3 k=1": perimage(b3_train.load_b3(m, "B3-k1"), D.dv)}
se = lambda d: d.std() / np.sqrt(len(d))
for k, v in K.items(): print(f"  {k:32s} {v.mean():.4f} (se {se(v):.4f})")
for a in ["formula, real moments (FITTED)", "formula, network-made moments"]:
    for b in ["trained B3 k=0", "trained B3 k=1"]:
        d = K[a] - K[b]; print(f"  {a} - {b}: {d.mean():+.4f} (se {se(d):.4f})")
np.savez("cache/r3e_paired.npz", **{k.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '').replace('=', ''): v for k, v in K.items()})
