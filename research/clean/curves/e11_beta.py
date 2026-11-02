"""E11 (H3): boundary step0 columns B with the constant beta recomputed from different input sets (plain means)."""
import numpy as np, torch
from inception import *
from family import Data
import w_fisher as W
from w_fisher import load_build, real_kl, intact, coarse, A_cols
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); M = load_build(m, fam, "L-k3-m-lr3e-2")
for p in m.parameters(): p.requires_grad_(False)
class Src:
    def __init__(self, X): self.X = X
BND = Src(np.load("cache/e7_bnd_1024_128_0.npy")[:512]); INV = Src(np.load("cache/e1_self_256.npy"))
c0, b0, _ = W.solve(m, fam, BND, M, np.arange(512), 10, 3); A = A_cols(m, fam)
def beta_from(src, idx):
    S = [0, 0]; Sz = [0, 0]; n = [0, 0]
    with torch.no_grad():
        for i in range(0, len(idx), 50):
            ins = inputs_3b(m, to_3a(m, prep(src.X[idx[i:i + 50]]))); _, z1, z2 = intact(m, fam, ins); c1, c2 = coarse(M, ins)
            for j, (z, c) in enumerate([(z1, c1), (z2, c2)]):
                Sz[j] = Sz[j] + z.sum((0, 2, 3)); S[j] = S[j] + c.sum((0, 2, 3)); n[j] += z.shape[0] * z.shape[2] * z.shape[3]
    return [A[j] @ (Sz[0 if j < 3 else 1] / n[0 if j < 3 else 1]) - c0[j] @ (S[0 if j < 3 else 1] / n[0 if j < 3 else 1]) for j in range(4)]
ridx = np.sort(np.random.default_rng(92).choice(D.tr, 512, replace=False))
B = {"boundary B, beta from Fisher solve (as used)": (M, c0, b0), "boundary B, plain-mean beta from boundary": (M, c0, beta_from(BND, np.arange(512))),
     "boundary B, beta from class inversions": (M, c0, beta_from(INV, np.arange(256))), "boundary B, beta from real (diagnostic)": (M, c0, beta_from(D, ridx))}
K = real_kl(m, fam, D, D.dv, B)
for k, v in K.items(): print(f"  {k:46s} dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})")
for j in range(4): print(f"  conv {j}: |beta| as used {float(b0[j].norm()):.1f}, boundary-mean {float(B['boundary B, plain-mean beta from boundary'][2][j].norm()):.1f}, real {float(B['boundary B, beta from real (diagnostic)'][2][j].norm()):.1f}")
