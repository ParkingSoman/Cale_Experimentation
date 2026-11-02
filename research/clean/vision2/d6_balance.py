"""D6 law-balanced partial cut. Cut set S (keep K). phi = sum_S gate / sum_all gate (signed net gate share, d6_roles).
Law: healing works when size leaves with the gate -> target r_S = 1 - phi (1 - r_full). Keep a fraction beta of the
block's constant bias, chosen by bisection so the median r on analysis images equals the target (a reading; no output
loss). Compare with bias kept (1) and dropped (0). Orders: energy-keep and |gate|-keep. usage: python d6_balance.py"""
import os, numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); b16, b17 = m.blocks[16], m.blocks[17]; gelu = nn.gelu
R = np.load(os.path.join(D, "d6_roles.npz")); gate, en = R["gate"], R["en"]
W1, c1 = np.array(b16.fc1.weight), np.array(b16.fc1.bias); W2 = np.array(b16.fc2.weight, np.float64) * np.array(b16.gamma, np.float64)[:, None]
b2 = np.array(b16.fc2.bias, np.float64) * np.array(b16.gamma, np.float64)
def keepf(K, beta):
    bb = mx.array((beta * b2).astype(np.float32)); A, cA, B = mx.array(W1[K].T), mx.array(c1[K]), mx.array(W2[:, K].T.astype(np.float32))
    return lambda x: x + gelu(b16.pre(x) @ A + cA) @ B + bb
sd = lambda v: np.sqrt(np.array(b17.conv_dw(v)).var(-1) + 1e-6)
XR = mx.array(z["XA"][::40]).astype(mx.float32); SI = sd(b16(XR))
rmed = lambda f: float(np.median(sd(f(XR)) / SI)); r_full = rmed(lambda x: x)
def sealed(f):
    T = mx.array(z["LE"]); out = []
    for i in range(0, 3000, 200):
        x = mx.array(z["XE"][i:i + 200]).astype(mx.float32); L = m.head(b17(f(x))); Tt = T[i:i + 200]
        lp = L - mx.logsumexp(L, 1, keepdims=True); tp = Tt - mx.logsumexp(Tt, 1, keepdims=True); out.append(np.array((mx.exp(tp) * (tp - lp)).sum(1)))
    return float(np.concatenate(out).mean())
print(f"r_full {r_full:.3f}  [full drop .1042]")
for name, order in (("energy-keep", np.argsort(-en)), ("|gate|-keep", np.argsort(-np.abs(gate)))):
    for k in (96, 384, 768, 1536, 2304):
        K = order[:k]; S = order[k:]; phi = gate[S].sum() / gate.sum(); tgt = 1 - phi * (1 - r_full)
        lo, hi = -1.0, 1.0
        for _ in range(14):
            mid = (lo + hi) / 2; (lo, hi) = (mid, hi) if rmed(keepf(K, mid)) < tgt else (lo, mid)
        beta = (lo + hi) / 2
        print(f"{name} k {k:4d}: phi {phi:.3f} target r {tgt:.3f} -> beta {beta:+.2f} | KL balanced {sealed(keepf(K, beta)):.4f} | bias kept {sealed(keepf(K, 1.0)):.4f} | bias dropped {sealed(keepf(K, 0.0)):.4f}", flush=True)
