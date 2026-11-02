"""R3f: Lambda from real vs network-made moments, row by row (direction cosine, norm ratio), plus level ratios."""
import numpy as np, torch
from inception import *
from family import pool3
m = load_model(); fam = list(np.load("cache/s1_fine.npz")["fam"]); keep = [i for i in range(480) if i not in fam]
X = np.load("cache/e1_self_256.npy", mmap_mode="r"); S = torch.zeros(481, 481, dtype=torch.float64)
with torch.no_grad():
    for i in range(0, len(X), 32):
        xb = np.asarray(X[i:i + 32])
        for xx in (xb, xb[:, :, ::-1]):
            x = pool3(F.relu(pre_3b(m, inputs_3b(m, to_3a(m, prep(np.ascontiguousarray(xx))))[0])))
            H = torch.cat([x.permute(0, 2, 3, 1).reshape(-1, 480), torch.ones(x.shape[0] * 196, 1, device=DEV)], 1).cpu().double()
            S += H.T @ H
np.save("cache/r3f_S3b_self.npy", S.numpy())
Sr = torch.from_numpy(np.load("cache/r3d_S3b.npy"))
def lam(S):
    R = keep + [480]; Srr = S[R][:, R]
    return torch.linalg.solve(Srr + 1e-6 * torch.trace(Srr) / len(R) * torch.eye(len(R), dtype=torch.float64), S[fam][:, R].T).T
Lr, Ls = lam(Sr / Sr[480, 480]), lam(S / S[480, 480])          # moments per position
cos = torch.nn.functional.cosine_similarity(Lr[:, :470], Ls[:, :470], 1); ratio = Ls[:, :470].norm(dim=1) / Lr[:, :470].norm(dim=1)
lev_r, lev_s = (Sr / Sr[480, 480]).diagonal()[:480], (S / S[480, 480]).diagonal()[:480]
print("Lambda rows (470 coefs): cos real vs network-made", np.round(cos.numpy(), 2))
print("norm ratio network-made / real", np.round(ratio.numpy(), 2))
print("constant term real", np.round(Lr[:, 470].numpy(), 3), "\n              made", np.round(Ls[:, 470].numpy(), 3))
lr_, ls_ = np.log(lev_r.numpy() + 1e-9), np.log(lev_s.numpy() + 1e-9)
print(f"3b levels E[h^2]: Spearman real vs made {__import__('scipy.stats').stats.spearmanr(lr_, ls_)[0]:.2f}; "
      f"median ratio made/real {np.median(np.exp(ls_ - lr_)):.2f}; within x2: {np.mean(np.abs(ls_ - lr_) < np.log(2)):.2f}")
Cr = (Sr / Sr[480, 480])[:480, :480]; Cs = (S / S[480, 480])[:480, :480]
nr = lambda C: C / torch.sqrt(C.diagonal()[:, None] * C.diagonal()[None, :] + 1e-12)
iu = torch.triu_indices(480, 480, 1)
print(f"normalised co-activation (480 x 480 off-diagonal): corr real vs made {np.corrcoef(nr(Cr)[iu[0], iu[1]].numpy(), nr(Cs)[iu[0], iu[1]].numpy())[0,1]:.3f}")
