"""D6 per-neuron affine law fix after dropping block 16. For each block-17 neuron j: z_I ~ alpha_j + beta_j z_D, fitted
(readings, analysis ranks 0-5) only on spots where the GELU responds: max(z_I, z_D) > T. Then block 17's fc1 row is
scaled by beta and its bias set accordingly (2 numbers per neuron). Arms: T = -1, -2, -3 (where 'responds' starts),
and a law-only version (no regression): beta = 1 on territory side, from the off-territory slope... (reported as the
regression's beta distribution). Score sealed KL. [plain drop .1042, best training .0836]"""
import os, copy, numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); b16, b17 = m.blocks[16], m.blocks[17]
rk = np.zeros(len(z["ana"]), int)
for c in range(1000): ix = np.where(z["yA"] == c)[0]; rk[ix] = np.arange(len(ix))
RD = np.where(rk < 6)[0]
S = {T: [0, 0, 0, 0, 0] for T in (-1.0, -2.0, -3.0)}           # n, sx, sy, sxx, sxy per neuron
for a in range(0, len(RD), 100):
    x = mx.array(z["XA"][RD[a:a + 100]]).astype(mx.float32)
    zi = np.array(b17.fc1(b17.pre(b16(x)))).reshape(-1, 3072).astype(np.float64); zd = np.array(b17.fc1(b17.pre(x))).reshape(-1, 3072).astype(np.float64)
    for T, s in S.items():
        w = (np.maximum(zi, zd) > T).astype(np.float64); s[0] += w.sum(0); s[1] += (w * zd).sum(0); s[2] += (w * zi).sum(0); s[3] += (w * zd * zd).sum(0); s[4] += (w * zd * zi).sum(0)
U0, c0 = np.array(b17.fc1.weight, np.float64), np.array(b17.fc1.bias, np.float64)
def sealed(beta, alpha):
    blk = copy.deepcopy(b17); blk.fc1.weight = mx.array((beta[:, None] * U0).astype(np.float32)); blk.fc1.bias = mx.array((beta * c0 + alpha).astype(np.float32))
    T = mx.array(z["LE"]); out = []
    for i in range(0, 3000, 200):
        L = m.head(blk(mx.array(z["XE"][i:i + 200]).astype(mx.float32))); Tt = T[i:i + 200]
        lp = L - mx.logsumexp(L, 1, keepdims=True); tp = Tt - mx.logsumexp(Tt, 1, keepdims=True); out.append(np.array((mx.exp(tp) * (tp - lp)).sum(1)))
    return float(np.concatenate(out).mean())
for T, (n, sx, sy, sxx, sxy) in S.items():
    ok = n > 50; mx_, my_ = sx / np.maximum(n, 1), sy / np.maximum(n, 1); vx = sxx / np.maximum(n, 1) - mx_ ** 2
    beta = np.where(ok & (vx > 1e-6), (sxy / np.maximum(n, 1) - mx_ * my_) / np.maximum(vx, 1e-6), 1.0); alpha = np.where(ok, my_ - beta * mx_, 0.0)
    print(f"T {T:+.0f}: beta median {np.median(beta[ok]):.3f} IQR {np.percentile(beta[ok], 25):.3f}-{np.percentile(beta[ok], 75):.3f}; alpha median {np.median(alpha[ok]):+.3f} | sealed KL {sealed(beta, alpha):.4f}", flush=True)
