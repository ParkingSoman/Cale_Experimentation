"""D6 robustness of the law slope for block 16. Same reader set (300 most affected, fixed from set A), two image sets:
A = cached validation ranks 18-19 (d5cache x16, float16), B = analysis rank 0 computed fresh in float32. Per neuron
slope of G on (1-r)(z_I-c) over ALL spots and over ON spots; median, IQR; pooled slope (all neurons together); and
bootstrap over neurons of the median."""
import os, numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
m = load_convnext(); X, y = load_images(); ST = m.stage_of; D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); b16, b17 = m.blocks[16], m.blocks[17]
rank = np.zeros(len(y), int)
for c in range(1000): ix = np.where(y == c)[0]; rank[ix] = np.arange(len(ix))
rk = np.zeros(len(z["ana"]), int)
for c in range(1000): ix = np.where(z["yA"] == c)[0]; rk[ix] = np.arange(len(ix))
U = np.array(b17.fc1.weight, np.float64); cc = U @ np.array(b17.norm.bias, np.float64) + np.array(b17.fc1.bias, np.float64)
def stream16(xs):
    x = m.stem_norm(m.stem(xs))
    for i in range(16):
        if i > 0 and ST[i] != ST[i - 1]: x = m.down[ST[i] - 1](m.down_norm[ST[i] - 1](x))
        x = m.blocks[i](x)
    return x
def collect(batches):
    ZI, ZD, R = [], [], []
    for x in batches:
        x1 = b16(x); ZI.append(np.array(b17.fc1(b17.pre(x1))).reshape(-1, 3072)); ZD.append(np.array(b17.fc1(b17.pre(x))).reshape(-1, 3072))
        sd = lambda v: np.sqrt(np.array(b17.conv_dw(v)).reshape(-1, 768).var(1) + 1e-6); R.append(sd(x) / sd(x1))
    ZI, ZD = np.concatenate(ZI).astype(np.float64), np.concatenate(ZD).astype(np.float64); r = np.concatenate(R).astype(np.float64)[:, None]
    G = ZI - cc - r * (ZD - cc); P = (1 - r) * (ZI - cc); return ZI, ZD, G, P
VA = np.where(rk >= 18)[0][::2]; A_ = collect(mx.array(z["XA"][VA[a:a + 100]]).astype(mx.float32) for a in range(0, len(VA), 100))
AN = np.where(rank == 0)[0][::2]; B_ = collect(stream16(mx.array(norm_img(np.asarray(X[AN[a:a + 50]])))) for a in range(0, len(AN), 50))
J = np.argsort(-np.abs(A_[0] - A_[1]).std(0) * np.linalg.norm(np.array(b17.fc2.weight), axis=0))[:300]
rng = np.random.default_rng(0)
for nm, (ZI, ZD, G, P) in (("A cached val 18-19", A_), ("B fresh analysis 0", B_)):
    for where in ("all", "on"):
        sl, gs, ps = [], [], []
        for j in J:
            s = np.ones(len(G), bool) if where == "all" else ZI[:, j] > 0
            if s.sum() < 20: continue
            g, p = G[s, j] - G[s, j].mean(), P[s, j] - P[s, j].mean(); sl.append((g @ p) / (p @ p + 1e-12)); gs.append(g); ps.append(p)
        sl = np.array(sl); g, p = np.concatenate(gs), np.concatenate(ps); boot = [np.median(rng.choice(sl, len(sl))) for _ in range(200)]
        print(f"{nm} {where:3s}: per-neuron slope median {np.median(sl):.3f} IQR {np.percentile(sl, 25):.3f}-{np.percentile(sl, 75):.3f} (boot 95% {np.percentile(boot, 2.5):.3f}-{np.percentile(boot, 97.5):.3f}) | pooled slope {(g @ p) / (p @ p):.3f} corr {np.corrcoef(g, p)[0, 1]:.3f}")
