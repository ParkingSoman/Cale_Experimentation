"""BV: bias vs variance of the network-made Fisher write (step0), measured in the real-data metric."""
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
X = np.load("cache/e15_aug.npy"); X = np.concatenate([X, X[:, :, ::-1]]); n = len(X); perm = np.random.default_rng(111).permutation(n)
P = softmax(np.asarray(D.T[D.tr], np.float64), 1); uns = D.tr[1 - (P ** 2).sum(1) > .1]; rng = np.random.default_rng(81)
r1 = np.sort(rng.choice(uns, 2000, replace=False)); r2 = np.sort(rng.choice(np.setdiff1d(uns, r1), 2000, replace=False))
for tag, src, idx in [("A", Src(X), np.sort(perm[:n // 2])), ("B", Src(X), np.sort(perm[n // 2:])), ("R1", D, r1), ("R2", D, r2)]:
    W.solve(m, fam, src, M, idx, 10, 3, save=f"cache/bv_{tag}.pt"); print("solved", tag, flush=True)
ld = lambda t: torch.load(f"cache/bv_{t}.pt")
Ns = {t: ld(t)["XtX"] / ld(t)["n"] for t in "A B R1 R2".split()}; ms = {t: ld(t)["Xty"] / ld(t)["n"] for t in Ns}; Pn = Ns["A"].shape[0]
sol = lambda N, mm: torch.linalg.solve(N + 1e-6 * torch.trace(N) / Pn * torch.eye(Pn, dtype=N.dtype), mm)
wA, wB, wR1, wR2 = sol(Ns["A"], ms["A"]), sol(Ns["B"], ms["B"]), sol(Ns["R1"], ms["R1"]), sol(Ns["R2"], ms["R2"])
wF = sol((Ns["A"] + Ns["B"]) / 2, (ms["A"] + ms["B"]) / 2); wR = sol((Ns["R1"] + Ns["R2"]) / 2, (ms["R1"] + ms["R2"]) / 2); NR = (Ns["R1"] + Ns["R2"]) / 2
q = lambda d: float(d @ NR @ d)
var_gen, var_real = q(wA - wB) / 4, q(wR1 - wR2) / 4; tot = q(wF - wR)
print(f"variance of the network-made write {var_gen:.4f}; of the real write {var_real:.4f}; total gap {tot:.4f}; bias = gap - both variances {tot - var_gen - var_real:.4f}")
print(f"for scale: real write energy {q(wR):.4f}")
