"""D5 vote gain: after dropping block 16, LayerNorm reads block 17's input ~1.36x louder (sigma ratio). Scale the
normalized part of every block-17 vote by k: votes = k * u.(lnw * n) + u.lnb + b. k = 1/mean ratio (reading) and others
(description)."""
import os, numpy as np, mlx.core as mx, copy
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); b16, b17 = m.blocks[16], m.blocks[17]
rk = np.zeros(len(z["ana"]), int)
for c in range(1000): ix = np.where(z["yA"] == c)[0]; rk[ix] = np.arange(len(ix))
x = mx.array(z["XA"][np.where(rk < 2)[0]]).astype(mx.float32); sd = lambda v: np.array(b17.conv_dw(v)).reshape(-1, 768).std(1)
ratio = sd(b16(x)) / sd(x); print(f"sigma_intact/sigma_dropped: mean {ratio.mean():.3f}; mean of inverse {np.mean(1 / ratio):.3f}")
U = np.array(b17.fc1.weight, np.float64); bb = np.array(b17.fc1.bias, np.float64); lb = np.array(b17.norm.bias, np.float64)
def sealed(k):
    blk = copy.deepcopy(b17); blk.fc1.weight = mx.array((k * U).astype(np.float32)); blk.fc1.bias = mx.array((bb + (1 - k) * U @ lb).astype(np.float32))
    T = mx.array(z["LE"]); out = []
    for a in range(0, 3000, 200):
        L = m.head(blk(mx.array(z["XE"][a:a + 200]).astype(mx.float32))); Tt = T[a:a + 200]
        lp = L - mx.logsumexp(L, 1, keepdims=True); tp = Tt - mx.logsumexp(Tt, 1, keepdims=True); out.append(np.array((mx.exp(tp) * (tp - lp)).sum(1)))
    return float(np.concatenate(out).mean())
k0 = 1 / ratio.mean()
for k in (1.0, 0.9, k0, 0.7, 1.1): print(f"  k {k:.3f}: sealed KL {sealed(k):.4f}" + ("   <- k = 1/ratio (reading)" if abs(k - k0) < 1e-9 else ""))
print("  [plain drop .1042, strong training .0942]")
