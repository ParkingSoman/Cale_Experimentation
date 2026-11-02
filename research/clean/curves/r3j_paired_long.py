"""R3j: paired dev-5000 of the real-moment formula against the long from-zero B3 k=0 ruler and the formula-start run."""
import numpy as np, torch
from inception import *
import b3_train
from family import Data, IN4A, kl
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); keep = [i for i in range(480) if i not in fam]
S = torch.from_numpy(np.load("cache/r3d_S3b.npy")); R = keep + [480]; Srr = S[R][:, R]; Sfr = S[fam][:, R]
Lam = torch.linalg.solve(Srr + 1e-6 * torch.trace(Srr) / len(R) * torch.eye(len(R), dtype=torch.float64), Sfr.T).T.float()
fm = b3_train.FamilyRA(m, fam, 1); fm.add_rb(m); fm.add_ra(m)
with torch.no_grad():
    for c in fm.cols: c.zero_()
    for j, n in enumerate(IN4A):
        w = getattr(m, n).weight.detach()[:, fam, 0, 0].cpu(); fm.dW[j].copy_((w @ Lam[:, :470]) / fm.cscale[j]); fm.cbp[j].copy_((w @ Lam[:, 470]) / 0.1)
def perimage(f, idx, bs=100):
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]; lg = b3_train.forward_b3(m, f, inputs_3b(m, to_3a(m, prep(D.X[b]))))
            out.append(kl(torch.from_numpy(D.T[b].astype(np.float32)).to(DEV), lg).cpu().numpy())
    return np.concatenate(out)
import sys
K = {"formula": perimage(fm.to(DEV), D.dv)}
for t in sys.argv[1:]: K[t] = perimage(b3_train.load_b3(m, t), D.dv)
for k, v in K.items():
    d = v - K["formula"]; print(f"{k:22s} {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})  minus formula {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})")
