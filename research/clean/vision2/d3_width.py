"""D3 width or depth? For block-16 neurons: best co-firing partner (corr of GELU outputs over positions) inside block 16
vs in blocks 15 / 17. 600 analysis images, 49 positions each."""
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
X, y = load_images(); m = load_convnext(); ST = m.stage_of; ana, _ = split(y); I = ana[::33]; H = {15: [], 16: [], 17: []}
for a in range(0, len(I), 50):
    x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(X[I[a:a + 50]])))))
    for b in range(18):
        if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
        if b in H: H[b].append(np.array(nn.gelu(m.blocks[b].fc1(m.blocks[b].pre(x)))).reshape(-1, 3072))
        x = m.blocks[b](x)
Z = {b: (lambda A: (A - A.mean(0)) / (A.std(0) + 1e-6))(np.concatenate(v).astype(np.float32)) for b, v in H.items()}; n = len(Z[16])
C = {b: Z[16].T @ Z[b] / n for b in H}; np.fill_diagonal(C[16], -1)
best = {b: C[b].max(1) for b in H}
for b in H: print(f"block 16 -> best partner in block {b}: median {np.median(best[b]):.3f}  p25 {np.percentile(best[b], 25):.3f}  p75 {np.percentile(best[b], 75):.3f}  share > .5: {(best[b] > .5).mean():.3f}")
w = best[16]; d = np.maximum(best[15], best[17]); print(f"share of neurons whose best partner is in another block: {(d > w).mean():.3f}; median (cross - within) {np.median(d - w):+.3f}")
# how many strong within-block partners per neuron (corr > .3)
print("within-block partners with corr > .3: median %d, mean %.1f; > .5: mean %.2f" % (np.median((C[16] > .3).sum(1)), (C[16] > .3).sum(1).mean(), (C[16] > .5).sum(1).mean()))
