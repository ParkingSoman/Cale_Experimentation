"""D6 novelty pruning of block 16 (derived from the proportionality law). Per block-16 neuron i, at its ON positions
(weighted by h_i): a_i = cos(write_i centered, stream s(p) centered) -> amplifier part; novelty energy
nov_i = E[h_i^2]^(1/2) |w_i| sqrt(1 - a_i^2). Keep the k neurons with the largest nov_i (others cut, NO mean added),
vs keeping the k largest by plain energy E[h^2]^(1/2)|w| (others cut, no mean), vs energy + mean (N10 style).
Readings on analysis ranks 0-2; score sealed KL. usage: python d6_novel.py"""
import os, numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); b16, b17 = m.blocks[16], m.blocks[17]; gelu = nn.gelu
rk = np.zeros(len(z["ana"]), int)
for c in range(1000): ix = np.where(z["yA"] == c)[0]; rk[ix] = np.arange(len(ix))
RD = np.where(rk < 3)[0]; W2 = np.array(b16.fc2.weight, np.float64) * np.array(b16.gamma, np.float64)[:, None]
Wc = W2 - W2.mean(0, keepdims=True); Wn = Wc / np.linalg.norm(Wc, axis=0, keepdims=True)
h2 = 0; hc = 0; hsum = 0; mh = 0; n = 0
for a in range(0, len(RD), 100):
    x = mx.array(z["XA"][RD[a:a + 100]]).astype(mx.float32); h = np.array(gelu(b16.fc1(b16.pre(x)))).reshape(-1, 3072).astype(np.float64)
    s = np.array(x).reshape(-1, 768).astype(np.float64); s = s - s.mean(1, keepdims=True); s /= np.linalg.norm(s, axis=1, keepdims=True)
    hp = np.maximum(h, 0); cosmat = s @ Wn                          # (positions, 3072): cos(write_i, stream at p)
    h2 += (h ** 2).sum(0); hc += (hp * cosmat).sum(0); hsum += hp.sum(0); mh += h.sum(0); n += h.shape[0]
a_ = hc / (hsum + 1e-12); en = np.sqrt(h2 / n) * np.linalg.norm(W2, axis=0); nov = en * np.sqrt(np.clip(1 - a_ ** 2, 0, 1)); mean_h = mh / n
print(f"amplifier cos a_i (write vs stream where the neuron fires): median {np.median(a_):.3f}, p10 {np.percentile(a_, 10):.3f}, p90 {np.percentile(a_, 90):.3f}; corr(a, energy) {np.corrcoef(a_, en)[0, 1]:.2f}")
print(f"energy-weighted mean a: {(a_ * en ** 2).sum() / (en ** 2).sum():.3f}")
W1, c1 = np.array(b16.fc1.weight), np.array(b16.fc1.bias); b2 = np.array(b16.fc2.bias, np.float64) * np.array(b16.gamma, np.float64)
def keepf(K, add_mean):
    L = np.setdiff1d(np.arange(3072), K); cst = (b2 + (W2[:, L] @ mean_h[L] if add_mean else 0)).astype(np.float32)
    A, cA, B = mx.array(W1[K].T), mx.array(c1[K]), mx.array(W2[:, K].T.astype(np.float32))
    return lambda x: x + gelu(b16.pre(x) @ A + cA) @ B + mx.array(cst)
def sealed(f):
    T = mx.array(z["LE"]); out = []
    for i in range(0, 3000, 200):
        x = mx.array(z["XE"][i:i + 200]).astype(mx.float32); L = m.head(b17(f(x))); Tt = T[i:i + 200]
        lp = L - mx.logsumexp(L, 1, keepdims=True); tp = Tt - mx.logsumexp(Tt, 1, keepdims=True); out.append(np.array((mx.exp(tp) * (tp - lp)).sum(1)))
    return float(np.concatenate(out).mean())
print("k kept | novelty-keep | energy-keep | energy-keep + mean of cut | amplifier-keep (control)")
for k in (96, 384, 768, 1536):
    print(f"{k:5d} | {sealed(keepf(np.argsort(-nov)[:k], False)):.4f} | {sealed(keepf(np.argsort(-en)[:k], False)):.4f} | {sealed(keepf(np.argsort(-en)[:k], True)):.4f} | {sealed(keepf(np.argsort(-(en * np.abs(a_)))[:k], False)):.4f}", flush=True)
print("plain drop (k=0) .1042")
