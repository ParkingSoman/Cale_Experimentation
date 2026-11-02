"""D6 proportionality law. Exact: lost vote z_D - z_I = [(1 - r)(z_I - c) - G] / r. Self-healing is exact when the gate G
is proportional to the final vote: G = (1 - r)(z_I - c). Per neuron (300 most affected), over ALL positions and over ON
positions: slope of G on (1 - r)(z_I - c) (law says 1), share of G's variance the proportional term explains, and the
size of the NOVELTY N = G - (1 - r)(z_I - c) (what block 16 adds that is not proportional). Also: is the novelty mostly
on ON positions or on off positions near threshold (where it decides firing)?"""
import os, numpy as np, mlx.core as mx
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); b16, b17 = m.blocks[16], m.blocks[17]
rk = np.zeros(len(z["ana"]), int)
for c in range(1000): ix = np.where(z["yA"] == c)[0]; rk[ix] = np.arange(len(ix))
VA = np.where(rk >= 18)[0][::2]; U = np.array(b17.fc1.weight, np.float64); cc = U @ np.array(b17.norm.bias, np.float64) + np.array(b17.fc1.bias, np.float64)
ZI, ZD, R = [], [], []
for a in range(0, len(VA), 100):
    x = mx.array(z["XA"][VA[a:a + 100]]).astype(mx.float32); x1 = b16(x)
    ZI.append(np.array(b17.fc1(b17.pre(x1))).reshape(-1, 3072)); ZD.append(np.array(b17.fc1(b17.pre(x))).reshape(-1, 3072))
    sd = lambda v: np.sqrt(np.array(b17.conv_dw(v)).reshape(-1, 768).var(1) + 1e-6); R.append(sd(x) / sd(x1))
ZI, ZD = np.concatenate(ZI).astype(np.float64), np.concatenate(ZD).astype(np.float64); r = np.concatenate(R).astype(np.float64)[:, None]
G = ZI - cc - r * (ZD - cc); P = (1 - r) * (ZI - cc); N = G - P
W2 = np.array(b17.fc2.weight) * np.array(b17.gamma)[:, None]; imp = np.linalg.norm(W2, axis=0) * np.abs(ZI - ZD).std(0); J = np.argsort(-imp)[:300]
out = []
for j in J:
    on = ZI[:, j] > 0; near = (ZI[:, j] > -2) & (ZI[:, j] <= 0)
    def st(s):
        g, p, n = G[s, j], P[s, j], N[s, j]; pc = p - p.mean(); return ((g - g.mean()) @ pc) / (pc @ pc + 1e-12), 1 - n.var() / (g.var() + 1e-12), np.abs(n).mean(), np.abs(g).mean()
    out.append(st(np.ones(len(G), bool)) + st(on) + st(near))
A = np.array(out); md = lambda k: np.median(A[:, k])
print(f"r median {np.median(r):.3f} -> proportional factor (1-r) median {np.median(1 - r):.3f}")
for nm, o in (("all positions", 0), ("ON positions", 4), ("near threshold (-2 < z_I <= 0)", 8)):
    print(f"  {nm:32s}: slope of G on (1-r)(z_I-c) {md(o):.3f} (law: 1) | variance of G explained {md(o + 1):.3f} | mean |novelty| {md(o + 2):.2f} vs mean |G| {md(o + 3):.2f}")
