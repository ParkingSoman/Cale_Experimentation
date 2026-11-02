"""C2 (grounding, images used only to CHECK). Theorem 2 premise: the trained net is a stationary point of
E[CE] + lam/2 ||theta||^2, so the mean loss gradient g = -lam * theta for every weight tensor.
Prediction (written first): cos(-g, theta) >= .5 for the 3b and 4a weights; lam_fit agrees across layers within 2x.
Split-half reliability of g is reported so noise is separated from failure. True labels, train split, + mirror."""
import numpy as np, torch, torch.nn.functional as F
from inception import *
from family import Data
m = load_model()
names = ["mixed3a_5x5_pre_relu_conv", "mixed3b_5x5_bottleneck_pre_relu_conv", "mixed3b_5x5_pre_relu_conv",
         "mixed3b_3x3_pre_relu_conv", "mixed3b_1x1_pre_relu_conv", "mixed4a_1x1_pre_relu_conv",
         "mixed4a_3x3_bottleneck_pre_relu_conv", "mixed4b_3x3_pre_relu_conv", "mixed5b_1x1_pre_relu_conv"]
P = {n: getattr(m, n).weight for n in names}
for p in P.values(): p.requires_grad_(True)
D = Data(); rng = np.random.default_rng(3)
idx = np.sort(np.concatenate([rng.choice(np.where(D.y[D.tr] == c)[0], 4, replace=False) for c in range(1000)]))
idx = D.tr[idx]; half = rng.permutation(len(idx)) % 2
G = [{n: torch.zeros_like(p) for n, p in P.items()} for _ in range(2)]; cnt = [0, 0]
bs = 40
for i in range(0, len(idx), bs):
    b = idx[i:i + bs]; hb = half[i:i + bs]
    for flip in [False, True]:
        x = prep(D.X[b]); x = torch.flip(x, [3]) if flip else x
        lg = m(x + 117.0) if False else None
        h = from_3b(m, F.relu(pre_3b(m, to_3a(m, x))))
        lab = torch.from_numpy(D.lab[D.y[b]]).to(DEV)
        for hh in [0, 1]:
            sel = torch.from_numpy(hb == hh).to(DEV)
            if sel.sum() == 0: continue
            loss = F.cross_entropy(h[sel], lab[sel], reduction="sum")
            gr = torch.autograd.grad(loss, list(P.values()), retain_graph=True)
            for n, g in zip(P, gr): G[hh][n] += g
            cnt[hh] += int(sel.sum())
    if i % 800 == 0: print(i, flush=True)
np.set_printoptions(precision=3, suppress=True)
fam = list(np.load("cache/s1_fine.npz")["fam"])
for n, p in P.items():
    g1, g2 = G[0][n] / cnt[0], G[1][n] / cnt[1]; g = (G[0][n] + G[1][n]) / sum(cnt); th = p.detach()
    cs = lambda a, b: float((a * b).sum() / (a.norm() * b.norm()))
    lam = float(-(g * th).sum() / (th * th).sum())
    print(f"{n:42s} cos(-g,theta) {cs(-g, th):+.3f} | split-half cos(g1,g2) {cs(g1, g2):+.3f} | halves vs theta {cs(-g1, th):+.3f} {cs(-g2, th):+.3f} | lam_fit {lam:.2e}")
    if n == "mixed3b_5x5_pre_relu_conv":
        r = [cs(-g[i - 320], th[i - 320]) for i in fam]
        print("   per curve kernel cos:", np.array(r))
