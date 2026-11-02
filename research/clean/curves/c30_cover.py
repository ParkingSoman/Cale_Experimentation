"""C30: coverage by multi-class uncertainty. Centroid blends of k class inversions; step0 Fisher write; coverage vs real."""
import numpy as np, torch
from inception import *
from family import Data
import w_fisher as W
from w_fisher import load_build, real_kl
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); M = load_build(m, fam, "L-k3-m-lr3e-2")
for p in m.parameters(): p.requires_grad_(False)
class Src:
    def __init__(self, X): self.X = X
INV = np.load("cache/e1_self_256.npy").astype(np.float32); yinv = np.load("cache/e1_self_256_y.npy")
Wl = m.softmax2_pre_activation_matmul.weight.detach()[torch.tensor(yinv)]; Wn = Wl / Wl.norm(dim=1, keepdim=True); S = (Wn @ Wn.T).cpu().numpy(); np.fill_diagonal(S, -2)
order = np.argsort(-S, 1)
def blends(ranks_list):
    X = [INV[np.r_[[a], order[a, r]]].mean(0) for r in ranks_list for a in range(256)]
    X = np.stack(X).clip(0, 255).astype(np.uint8); return np.concatenate([X, X[:, :, ::-1]])
sets = {"2-way": blends([[0], [1]]), "4-way": blends([[0, 1, 2], [1, 2, 3]])}
Nr = torch.load("cache/e10_N_real.pt"); Nr = Nr["XtX"] / Nr["n"]; P = Nr.shape[0]; out = {}
for k, X in sets.items():
    c0, b0, _ = W.solve(m, fam, Src(X), M, np.arange(len(X)), 10, 3, save=f"cache/c30_N_{k}.pt"); out[k] = (M, c0, b0)
    d = torch.load(f"cache/c30_N_{k}.pt"); Nq = d["XtX"] / d["n"]; L = torch.linalg.cholesky(Nq + 1e-6 * torch.trace(Nq) / P * torch.eye(P, dtype=Nq.dtype)); Li = torch.linalg.inv(L)
    ev = torch.linalg.eigvalsh(Li @ Nr @ Li.T).flip(0)
    with torch.no_grad():
        lg = []
        for i in range(0, 256, 32):
            x = prep(X[i:i + 32]); a3 = to_3a(m, x)
        p = None
    print(f"{k}: generalised eigenvalues vs real-unsure 512: top {float(ev[0]):.3e}, 10th {float(ev[9]):.3e}, 100th {float(ev[99]):.3e}", flush=True)
K = real_kl(m, fam, D, D.dv, out)
for k, v in K.items(): print(f"  step0 from {k} blends (1024): dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})")
