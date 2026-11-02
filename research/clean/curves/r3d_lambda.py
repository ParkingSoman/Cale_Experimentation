"""R3d (Prop. 11.1 premise check, FITTED moments): Delta-W = W_F Lambda, Lambda = S_F,rest S_rest,rest^-1 from the pooled
3b activity second moments (with a constant), 2000 train images. Readers re-route through the 470 others; no coarse unit."""
import numpy as np, torch
from inception import *
import b3_train
from family import Data, evaluate, IN4A, pool3
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"])
keep = [i for i in range(480) if i not in fam]
tr = np.sort(np.random.default_rng(5).choice(D.tr, 2000, replace=False))
S = torch.zeros(481, 481, dtype=torch.float64)
with torch.no_grad():
    for i in range(0, len(tr), 50):
        x = pool3(F.relu(pre_3b(m, inputs_3b(m, to_3a(m, prep(D.X[tr[i:i + 50]])))[0])))
        H = torch.cat([x.permute(0, 2, 3, 1).reshape(-1, 480), torch.ones(x.shape[0] * 196, 1, device=DEV)], 1).cpu().double()
        S += H.T @ H
np.save("cache/r3d_S3b.npy", S.numpy())
R = keep + [480]
Srr = S[R][:, R]; Sfr = S[fam][:, R]
Lam = torch.linalg.solve(Srr + 1e-6 * torch.trace(Srr) / len(R) * torch.eye(len(R), dtype=torch.float64), Sfr.T).T.float()   # (10, 471)
WF = [getattr(m, n).weight.detach()[:, fam, 0, 0].cpu() for n in IN4A]
fm = b3_train.FamilyRA(m, fam, 1); fm.add_rb(m); fm.add_ra(m)
with torch.no_grad():
    for c in fm.cols: c.zero_()
    for j, (w, s) in enumerate(zip(WF, fm.cscale)):
        fm.dW[j].copy_((w @ Lam[:, :470]) / s); fm.cbp[j].copy_((w @ Lam[:, 470]) / 0.1)
fm = fm.to(DEV)
print("Lambda (FITTED) re-routing, no coarse unit: dev-5000", evaluate(m, fm, D, D.dv), flush=True)
r2 = 1 - (S[fam][:, fam].diagonal() - (Lam.double() @ Sfr.T).diagonal()) / S[fam][:, fam].diagonal()
print("share of each curve neuron's second moment explained by the rest:", np.round(r2.numpy(), 2))
