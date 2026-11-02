"""E10: why the boundary Fisher write blows up. Spectrum of N (step0 normal matrix) and step0 KL, boundary vs real unsure,
512 inputs each, M's read."""
import numpy as np, torch
from inception import *
from family import Data
import w_fisher as W
from w_fisher import load_build, real_kl
from scipy.special import softmax
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); M = load_build(m, fam, "L-k3-m-lr3e-2")
for p in m.parameters(): p.requires_grad_(False)
class Src:
    def __init__(self, X): self.X = X
Bx = np.load("cache/e7_bnd_1024_128_0.npy")[:512]
P = softmax(np.asarray(D.T[D.tr], np.float64), 1); uns = D.tr[1 - (P ** 2).sum(1) > .1]; ridx = np.sort(np.random.default_rng(91).choice(uns, 512, replace=False))
out = {}
for tag, src, idx in [("boundary", Src(Bx), np.arange(512)), ("real unsure", D, ridx)]:
    c0, b0, cov = W.solve(m, fam, src, M, idx, 10, 3, save=f"cache/e10_N_{tag.split()[0]}.pt"); out[tag] = (M, c0, b0)
K = real_kl(m, fam, D, D.dv, out)
for k, v in K.items(): print(f"  step0 from {k:12s} (512): dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})")
for tag in ["boundary", "real"]:
    d = torch.load(f"cache/e10_N_{tag}.pt"); N = d["XtX"] / d["n"]; ev = torch.linalg.eigvalsh(N).flip(0).clamp(min=0)
    print(f"  N spectrum {tag:8s}: top {float(ev[0]):.3e}; eigenvalues above 1e-3 x top: {int((ev > 1e-3 * ev[0]).sum())}, above 1e-5 x top: {int((ev > 1e-5 * ev[0]).sum())} of {len(ev)}; trace {float(ev.sum()):.3e}")
Nb, Nr = [torch.load(f"cache/e10_N_{t}.pt") for t in ["boundary", "real"]]; Nb, Nr = Nb["XtX"] / Nb["n"], Nr["XtX"] / Nr["n"]
e, U = torch.linalg.eigh(Nr); top = U[:, -200:]   # the real data's 200 strongest write directions
cover = torch.diag(top.T @ Nb @ top) / torch.diag(top.T @ Nr @ top)
print(f"  boundary mass / real mass along the real top-200 directions: median {float(cover.median()):.3e}, 10th pct {float(cover.quantile(.1)):.3e}")
