"""D6 Bussgang coarse block. Each block-16 neuron i: GELU(z_i) ~ a_i + b_i z_i with b_i = E[GELU'(z_i)] (Bussgang gain),
a_i = E[GELU(z_i)] - b_i E[z_i] (readings on analysis images; no fitting). A linearized neuron folds into one 768x768 map:
  block16(x) = x + gamma*(W2[:,K] gelu(z_K) + W2[:,L](a_L + b_L z_L) + b2)    K = kept neurons, L = linearized ones
Dial: keep the k neurons whose nonlinearity matters most, score = |write| * std(GELU(z) - (a + b z)) (nonlinear residue).
Score: sealed KL of the whole model. usage: python d6_bussgang.py"""
import os, numpy as np, mlx.core as mx, mlx.nn as nn, copy
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); b16, b17 = m.blocks[16], m.blocks[17]; gelu = nn.gelu
rk = np.zeros(len(z["ana"]), int)
for c in range(1000): ix = np.where(z["yA"] == c)[0]; rk[ix] = np.arange(len(ix))
RD = np.where(rk < 3)[0]
def gd(v): return 0.5 * (1 + mx.erf(v / 2 ** .5)) + v * mx.exp(-v * v / 2) / (2 * np.pi) ** .5
S = dict(g=0, gp=0, z=0, h=0, n=0); Zs = []
for a in range(0, len(RD), 100):
    x = mx.array(z["XA"][RD[a:a + 100]]).astype(mx.float32); zz = b16.fc1(b16.pre(x)).reshape(-1, 3072)
    S["gp"] += np.array(gd(zz).sum(0)); S["z"] += np.array(zz.sum(0)); S["h"] += np.array(gelu(zz).sum(0)); S["n"] += zz.shape[0]
    if a < 1000: Zs.append(np.array(zz))
b = S["gp"] / S["n"]; a_ = S["h"] / S["n"] - b * S["z"] / S["n"]; Zs = np.concatenate(Zs).astype(np.float64)
res = np.array(gelu(mx.array(Zs.astype(np.float32)))) - (a_ + b * Zs); W2 = np.array(b16.fc2.weight) * np.array(b16.gamma)[:, None]
score = np.linalg.norm(W2, axis=0) * res.std(0); order = np.argsort(-score)
print(f"Bussgang gains b: median {np.median(b):.3f}, p10 {np.percentile(b, 10):.3f}, p90 {np.percentile(b, 90):.3f}; nonlinear residue share of output var: {(res.var(0) * (W2 ** 2).sum(0)).sum() / (np.array(gelu(mx.array(Zs.astype(np.float32)))).var(0) * (W2 ** 2).sum(0)).sum():.3f}")
W1, c1 = np.array(b16.fc1.weight, np.float64), np.array(b16.fc1.bias, np.float64)
def coarse(k):
    K, L = order[:k], order[k:]
    M = (W2[:, L] * b[L]) @ W1[L]; cst = W2[:, L] @ (a_[L] + b[L] * c1[L]) + np.array(b16.fc2.bias) * np.array(b16.gamma)
    def f(x):
        pre = b16.pre(x); out = mx.array(cst.astype(np.float32)) + pre @ mx.array(M.T.astype(np.float32))
        if k: out = out + gelu(pre @ mx.array(W1[K].T.astype(np.float32)) + mx.array(c1[K].astype(np.float32))) @ mx.array(W2[:, K].T.astype(np.float32))
        return x + out
    return f
def sealed(f16):
    T = mx.array(z["LE"]); out = []
    for i in range(0, 3000, 200):
        x = mx.array(z["XE"][i:i + 200]).astype(mx.float32); L = m.head(b17(f16(x))); Tt = T[i:i + 200]
        lp = L - mx.logsumexp(L, 1, keepdims=True); tp = Tt - mx.logsumexp(Tt, 1, keepdims=True); out.append(np.array((mx.exp(tp) * (tp - lp)).sum(1)))
    return float(np.concatenate(out).mean())
print(f"drop block 16 {sealed(lambda x: x):.4f}  |  intact {sealed(b16):.4f}")
for k in (0, 64, 256, 768, 1536):
    print(f"  keep {k:4d} real neurons + {3072 - k} Bussgang-linearized (extra cost 768x768): sealed KL {sealed(coarse(k)):.4f}", flush=True)
