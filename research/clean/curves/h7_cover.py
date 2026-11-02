"""H7: coverage baseline (real vs real) and the composition of the remaining worst directions."""
import numpy as np, torch
from inception import *
from family import Data
import w_fisher as W
from w_fisher import load_build
from scipy.special import softmax
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); M = load_build(m, fam, "L-k3-m-lr3e-2")
for p in m.parameters(): p.requires_grad_(False)
class Src:
    def __init__(self, X): self.X = X
P = softmax(np.asarray(D.T[D.tr], np.float64), 1); uns = D.tr[1 - (P ** 2).sum(1) > .1]
r1 = np.sort(np.random.default_rng(91).choice(uns, 512, replace=False)); r2 = np.sort(np.random.default_rng(94).choice(np.setdiff1d(uns, r1), 512, replace=False))
W.solve(m, fam, D, M, r2, 10, 3, save="cache/h7_N_real2.pt")
X = np.load("cache/e14_set.npy"); X = np.concatenate([X, X[:, :, ::-1]]); W.solve(m, fam, Src(X), M, np.arange(len(X)), 10, 3, save="cache/h7_N_comb.pt")
ld = lambda f: (lambda d: d["XtX"] / d["n"])(torch.load(f))
N1, N2, Nc = ld("cache/e10_N_real.pt"), ld("cache/h7_N_real2.pt"), ld("cache/h7_N_comb.pt"); Pn = N1.shape[0]
def gen(Nt, Nq):
    L = torch.linalg.cholesky(Nq + 1e-6 * torch.trace(Nq) / Pn * torch.eye(Pn, dtype=Nq.dtype)); Li = torch.linalg.inv(L)
    ev, V = torch.linalg.eigh(Li @ Nt @ Li.T); return ev.flip(0), (Li.T @ V).flip(1)
ev, _ = gen(N1, N2); print(f"real(512) vs independent real(512): top {float(ev[0]):.3e}, 10th {float(ev[9]):.3e}, 100th {float(ev[99]):.3e}")
ev, dirs = gen(N1, Nc); print(f"real(512) vs combined network-made: top {float(ev[0]):.3e}, 10th {float(ev[9]):.3e}, 100th {float(ev[99]):.3e}")
d = dirs[:, :20] / dirs[:, :20].norm(dim=0); w = (d ** 2).reshape(368, 4, 20).sum(2)
print(f"top-20 directions: share on unit columns {float(w[:, :3].sum() / w.sum()):.3f} (per unit {(w[:, :3].sum(0) / w.sum()).numpy().round(3)}), on the constant {float(w[:, 3].sum() / w.sum()):.3f}")
