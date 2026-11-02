"""E12 (H4): size and direction of the step0 columns from boundary vs real unsure (512 each); M's trained write for scale."""
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
P = softmax(np.asarray(D.T[D.tr], np.float64), 1); uns = D.tr[1 - (P ** 2).sum(1) > .1]; ridx = np.sort(np.random.default_rng(91).choice(uns, 512, replace=False))
cb, _, _ = W.solve(m, fam, Src(np.load("cache/e7_bnd_1024_128_0.npy")[:512]), M, np.arange(512), 10, 3)
cr, _, _ = W.solve(m, fam, D, M, ridx, 10, 3); ct = [c for c in M.weights()[2]]
flat = lambda cs: torch.cat([c.flatten() for c in cs])
for u in range(3):
    b, r, t = [torch.cat([c[:, u] for c in cs]) for cs in (cb, cr, ct)]
    print(f"unit {u}: |B_bnd| {float(b.norm()):.4f} |B_real| {float(r.norm()):.4f} |B_trained| {float(t.norm()):.4f} | cos(bnd, real) {float(F.cosine_similarity(b, r, 0)):+.3f} cos(real, trained) {float(F.cosine_similarity(r, t, 0)):+.3f} cos(bnd, trained) {float(F.cosine_similarity(b, t, 0)):+.3f}")
with torch.no_grad():
    for tag, X in [("boundary", np.load("cache/e7_bnd_1024_128_0.npy")[:64]), ("real unsure", D.X[ridx[:64]])]:
        ins = inputs_3b(m, to_3a(m, prep(X))); c = F.relu(M.pre(ins)); z = F.relu(pre_3b(m, ins[0])[:, fam])
        print(f"  {tag:11s}: coarse units mean {c.mean((0, 2, 3)).cpu().numpy().round(1)}, on-rate {(c > 0).float().mean((0, 2, 3)).cpu().numpy().round(3)} | family mean {float(z.mean()):.1f}")
