"""D4 check 2: how much of block B's votes can block B+1's pre (dwconv+LN of the SAME stream) carry? Conditional
expectation (linear): z_i ~ v_i . pre_{B+1} + c_i, v from covariances on half the images, R^2 on the other half."""
import sys, numpy as np, mlx.core as mx
from convnext import *
B = int(sys.argv[1]); X, y = load_images(); m = load_convnext(); ST = m.stage_of; ana, _ = split(y); I = ana[::33]
b0, b1_ = m.blocks[B], m.blocks[B + 1]; W1 = np.array(b0.fc1.weight, np.float64); c1 = np.array(b0.fc1.bias, np.float64)
Zt, P1 = [], []
for a in range(0, len(I), 50):
    x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(X[I[a:a + 50]])))))
    for b in range(B):
        if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
        x = m.blocks[b](x)
    if B > 0 and ST[B] != ST[B - 1]: x = m.down[ST[B] - 1](m.down_norm[ST[B] - 1](x))
    Zt.append(np.array(b0.pre(x)).reshape(-1, W1.shape[1]).astype(np.float64) @ W1.T + c1); P1.append(np.array(b1_.pre(x)).reshape(-1, W1.shape[1]).astype(np.float64))
Zt, P1 = np.concatenate(Zt), np.concatenate(P1); R = Zt.shape[0] // len(I); half = np.repeat(np.arange(len(I)) % 2 == 0, R)
A = np.c_[P1, np.ones(len(P1))]; V = np.linalg.lstsq(A[half], Zt[half], rcond=None)[0]; E = A[~half] @ V - Zt[~half]
r2 = 1 - (E ** 2).mean(0) / Zt[~half].var(0); on = (Zt > 0).mean(0) > .01
print(f"block {B} votes from block {B + 1}'s pre (same stream), held-out R^2: median {np.median(r2):.3f} p10 {np.percentile(r2, 10):.3f} p90 {np.percentile(r2, 90):.3f}; alive neurons median {np.median(r2[on]):.3f}")
