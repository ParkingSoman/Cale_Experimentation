"""D4 check: if block B is removed, block B+1 reads the stream block B used to read. Can block B's neurons be computed
with block B+1's dwconv + LayerNorm (read transplanted: u' = u * lnw_B / lnw_{B+1}, bias fixed for the LN biases)?
Corr of the transplanted vote with the true vote, per neuron. 300 images."""
import sys, numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
B = int(sys.argv[1]); X, y = load_images(); m = load_convnext(); ST = m.stage_of; ana, _ = split(y); I = ana[::66]
b0, b1_ = m.blocks[B], m.blocks[B + 1]; W1 = np.array(b0.fc1.weight, np.float64); c1 = np.array(b0.fc1.bias, np.float64)
g0, be0, g1, be1 = (np.array(v, np.float64) for v in (b0.norm.weight, b0.norm.bias, b1_.norm.weight, b1_.norm.bias))
Wt = W1 * (g0 / g1)[None]; ct = c1 + W1 @ (be0 - be1 * g0 / g1)
Zt, Zx = [], []
for a in range(0, len(I), 50):
    x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(X[I[a:a + 50]])))))
    for b in range(B):
        if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
        x = m.blocks[b](x)
    if B > 0 and ST[B] != ST[B - 1]: x = m.down[ST[B] - 1](m.down_norm[ST[B] - 1](x))
    p0 = np.array(b0.pre(x)).reshape(-1, W1.shape[1]).astype(np.float64); p1 = np.array(b1_.pre(x)).reshape(-1, W1.shape[1]).astype(np.float64)
    Zt.append(p0 @ W1.T + c1); Zx.append(p1 @ Wt.T + ct)
Zt, Zx = np.concatenate(Zt), np.concatenate(Zx)
c = np.array([np.corrcoef(Zt[:, i], Zx[:, i])[0, 1] for i in range(W1.shape[0])])
print(f"block {B} reads computed through block {B + 1}'s dwconv+LN: vote corr median {np.median(c):.3f} p10 {np.percentile(c, 10):.3f} p90 {np.percentile(c, 90):.3f}")
# dwconv kernels: how similar are the two blocks' per-channel kernels?
k0 = np.array(b0.conv_dw.weight).reshape(768 if W1.shape[1] == 768 else W1.shape[1], -1); k1 = np.array(b1_.conv_dw.weight).reshape(k0.shape[0], -1)
kc = (k0 * k1).sum(1) / np.linalg.norm(k0, axis=1) / np.linalg.norm(k1, axis=1); print(f"per-channel dwconv kernel cos B vs B+1: median {np.median(kc):.3f}")
