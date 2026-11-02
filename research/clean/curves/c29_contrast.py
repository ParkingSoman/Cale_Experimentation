"""C29: Fisher step0 write from real unsure images at contrast x1 vs x.5 (512 each, same images)."""
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
P = softmax(np.asarray(D.T[D.tr], np.float64), 1); uns = D.tr[1 - (P ** 2).sum(1) > .1]; ridx = np.sort(np.random.default_rng(91).choice(uns, 512, replace=False))
R = np.asarray(D.X[ridx]).astype(np.float32); out = {}; cols = {}
for s in [1.0, 0.5]:
    X = np.clip(117 + s * (R - 117), 0, 255).astype(np.uint8); c0, b0, _ = W.solve(m, fam, Src(X), M, np.arange(512), 10, 3); out[f"contrast x{s}"] = (M, c0, b0); cols[s] = c0
for u in range(3):
    a, b = [torch.cat([c[:, u] for c in cols[s]]) for s in (1.0, 0.5)]; print(f"unit {u}: cos(x1, x.5) {float(F.cosine_similarity(a, b, 0)):+.3f}, norm ratio {float(b.norm() / a.norm()):.2f}")
K = real_kl(m, fam, D, D.dv, out)
for k, v in K.items(): print(f"  step0 from real unsure {k}: dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})")
