"""D6 is the block's constant output bias (fc2 bias x gamma) what kills the healing in partial cuts?
Arms: full drop; full drop + bias only; keep-k (energy) with bias; keep-k without bias. Also: bias size vs block output,
and how much of the bias lies along the least-read stream direction."""
import os, numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); b16, b17 = m.blocks[16], m.blocks[17]; gelu = nn.gelu
R = np.load(os.path.join(D, "d6_roles.npz")); en = R["en"]
W1, c1 = np.array(b16.fc1.weight), np.array(b16.fc1.bias); W2 = np.array(b16.fc2.weight, np.float64) * np.array(b16.gamma, np.float64)[:, None]
b2 = np.array(b16.fc2.bias, np.float64) * np.array(b16.gamma, np.float64)
x = mx.array(z["XA"][:300]).astype(mx.float32); O = np.array(b16(x) - x).reshape(-1, 768).astype(np.float64); S = np.array(x).reshape(-1, 768).astype(np.float64)
cen = lambda v: v - v.mean(-1, keepdims=True)
print(f"|bias (centered)| {np.linalg.norm(cen(b2)):.2f}; mean |block-16 output (centered)| {np.linalg.norm(cen(O), axis=1).mean():.2f}; mean |stream (centered)| {np.linalg.norm(cen(S), axis=1).mean():.2f}")
print(f"cos(bias, mean block output) {cen(b2) @ cen(O.mean(0)) / np.linalg.norm(cen(b2)) / np.linalg.norm(cen(O.mean(0))):.3f}; cos(bias, mean stream) {cen(b2) @ cen(S.mean(0)) / np.linalg.norm(cen(b2)) / np.linalg.norm(cen(S.mean(0))):.3f}")
def keepf(K, bias):
    bb = mx.array(b2.astype(np.float32)) if bias else 0
    if len(K) == 0: return lambda x: x + bb
    A, cA, B = mx.array(W1[K].T), mx.array(c1[K]), mx.array(W2[:, K].T.astype(np.float32)); return lambda x: x + gelu(b16.pre(x) @ A + cA) @ B + bb
sd = lambda v: np.sqrt(np.array(b17.conv_dw(v)).var(-1) + 1e-6)
def sealed(f):
    T = mx.array(z["LE"]); out = []; rr = []
    for i in range(0, 3000, 200):
        x = mx.array(z["XE"][i:i + 200]).astype(mx.float32); xc = f(x); L = m.head(b17(xc)); Tt = T[i:i + 200]
        lp = L - mx.logsumexp(L, 1, keepdims=True); tp = Tt - mx.logsumexp(Tt, 1, keepdims=True); out.append(np.array((mx.exp(tp) * (tp - lp)).sum(1)))
        if i < 600: rr.append((sd(xc) / sd(b16(x))).ravel())
    return f"{float(np.concatenate(out).mean()):.4f} (r {float(np.median(np.concatenate(rr))):.3f})"
print("full drop:", sealed(keepf([], False)), "| full drop + bias only:", sealed(keepf([], True)))
o = np.argsort(-en)
for k in (32, 96, 384, 768, 1536, 2304): print(f"keep {k:4d} by energy: with bias {sealed(keepf(o[:k], True))} | without bias {sealed(keepf(o[:k], False))}", flush=True)
