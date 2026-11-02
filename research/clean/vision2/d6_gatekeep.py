"""D6 law prediction: keep the k block-16 neurons with the largest |gate| share (little volume), drop the rest (no mean).
Healing (stream shrink) should survive and part of the gate is kept -> better than the full drop. Compare: energy-keep,
gate-keep, gate-per-volume keep (|gate| / max(volume, small)). Reports r (size ratio) too."""
import os, numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); b16, b17 = m.blocks[16], m.blocks[17]; gelu = nn.gelu
R = np.load(os.path.join(D, "d6_roles.npz")); vol, gate, en = R["vol"], R["gate"], R["en"]
W1, c1 = np.array(b16.fc1.weight), np.array(b16.fc1.bias); W2 = np.array(b16.fc2.weight, np.float64) * np.array(b16.gamma, np.float64)[:, None]
b2 = (np.array(b16.fc2.bias, np.float64) * np.array(b16.gamma, np.float64)).astype(np.float32)
def keepf(K):
    A, cA, B = mx.array(W1[K].T), mx.array(c1[K]), mx.array(W2[:, K].T.astype(np.float32)); return lambda x: x + gelu(b16.pre(x) @ A + cA) @ B + mx.array(b2)
sd = lambda v: np.sqrt(np.array(b17.conv_dw(v)).var(-1) + 1e-6)
def sealed(f):
    T = mx.array(z["LE"]); out = []; rr = []
    for i in range(0, 3000, 200):
        x = mx.array(z["XE"][i:i + 200]).astype(mx.float32); xc = f(x); L = m.head(b17(xc)); Tt = T[i:i + 200]
        lp = L - mx.logsumexp(L, 1, keepdims=True); tp = Tt - mx.logsumexp(Tt, 1, keepdims=True); out.append(np.array((mx.exp(tp) * (tp - lp)).sum(1)))
        if i < 600: rr.append((sd(xc) / sd(b16(x))).ravel())
    return float(np.concatenate(out).mean()), float(np.median(np.concatenate(rr)))
ratio = np.abs(gate) / np.maximum(vol, 1e-4)
print("k | energy-keep KL (r) | |gate|-keep KL (r) | gate-per-volume keep KL (r)   [full drop .1042, r .735]")
for k in (32, 96, 384, 768):
    a, b, c = sealed(keepf(np.argsort(-en)[:k])), sealed(keepf(np.argsort(-np.abs(gate))[:k])), sealed(keepf(np.argsort(-ratio)[:k]))
    print(f"{k:4d} | {a[0]:.4f} ({a[1]:.3f}) | {b[0]:.4f} ({b[1]:.3f}) | {c[0]:.4f} ({c[1]:.3f})", flush=True)
