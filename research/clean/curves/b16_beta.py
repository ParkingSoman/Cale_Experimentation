"""B16: network-made step0 columns with beta from different sources."""
import numpy as np, torch
from inception import *
from family import Data
from w_fisher import load_build, real_kl, intact, coarse, A_cols
from scipy.special import softmax
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); M = load_build(m, fam, "L-k3-m-lr3e-2")
for p in m.parameters(): p.requires_grad_(False)
class Src:
    def __init__(self, X): self.X = X
ld = lambda t: torch.load(f"cache/bv_{t}.pt"); N = (ld("A")["XtX"] + ld("B")["XtX"]) / (ld("A")["n"] + ld("B")["n"]); mm = (ld("A")["Xty"] + ld("B")["Xty"]) / (ld("A")["n"] + ld("B")["n"])
Pn = N.shape[0]; w = torch.linalg.solve(N + 1e-6 * torch.trace(N) / Pn * torch.eye(Pn, dtype=N.dtype), mm).float(); A = A_cols(m, fam); sizes = [a.shape[0] for a in A]
cols, cbs, o = [], [], 0
for s in sizes:
    Wt = w[o:o + 4 * s].reshape(s, 4); o += 4 * s; cols.append(Wt[:, :3].to(DEV)); cbs.append(Wt[:, 3].to(DEV))
def beta_from(src, idx):
    S, Sz, n = [0, 0], [0, 0], [0, 0]
    with torch.no_grad():
        for i in range(0, len(idx), 50):
            ins = inputs_3b(m, to_3a(m, prep(src.X[idx[i:i + 50]]))); _, z1, z2 = intact(m, fam, ins); c1, c2 = coarse(M, ins)
            for j, (z, c) in enumerate([(z1, c1), (z2, c2)]): Sz[j] = Sz[j] + z.sum((0, 2, 3)); S[j] = S[j] + c.sum((0, 2, 3)); n[j] += z.shape[0] * z.shape[2] * z.shape[3]
    return [A[j] @ (Sz[min(j // 3, 1)] / n[min(j // 3, 1)]) - cols[j] @ (S[min(j // 3, 1)] / n[min(j // 3, 1)]) for j in range(4)]
P = softmax(np.asarray(D.T[D.tr], np.float64), 1); uns = D.tr[1 - (P ** 2).sum(1) > .1]; r = np.sort(np.random.default_rng(81).choice(uns, 2000, replace=False))
AUG = Src(np.load("cache/e15_aug.npy")); INV = Src(np.load("cache/e1_self_256.npy"))
B = {"network-made B, solve's beta": (M, cols, cbs), "network-made B, beta from inversions": (M, cols, beta_from(INV, np.arange(256))),
     "network-made B, plain-mean beta from the set": (M, cols, beta_from(AUG, np.arange(0, len(AUG.X), 4))), "network-made B, beta from real (diagnostic)": (M, cols, beta_from(D, r))}
K = real_kl(m, fam, D, D.dv, B)
for k, v in K.items(): print(f"  {k:46s} dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})")
