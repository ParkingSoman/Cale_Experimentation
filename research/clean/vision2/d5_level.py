"""D5 the level clue: across all block-17 neurons, regress (trained - dropped) on (intact - dropped) for: mean activation
over positions, firing rate (z > 0), mean vote z. Slope ~ 1 -> training restores each neuron's level."""
import os, sys, copy
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); gelu = nn.gelu
rk = np.zeros(len(z["ana"]), int)
for c in range(1000): ix = np.where(z["yA"] == c)[0]; rk[ix] = np.arange(len(ix))
VA = np.where(rk >= 18)[0]; b16, b17 = m.blocks[16], m.blocks[17]; t17 = copy.deepcopy(b17); t17.load_weights([(k, mx.array(v)) for k, v in np.load(sys.argv[1]).items()])
S = {k: [0, 0, 0] for k in "idt"}; n = 0
for a in range(0, len(VA), 100):
    x = mx.array(z["XA"][VA[a:a + 100]]).astype(mx.float32)
    for k, blk, xx in (("i", b17, b16(x)), ("d", b17, x), ("t", t17, x)):
        zz = np.array(blk.fc1(blk.pre(xx))).reshape(-1, 3072).astype(np.float64); h = np.array(gelu(mx.array(zz.astype(np.float32)))).astype(np.float64)
        S[k][0] += h.sum(0); S[k][1] += (zz > 0).sum(0); S[k][2] += zz.sum(0)
    n += x.shape[0] * 49
for q, nm in enumerate(("mean activation", "firing rate", "mean vote")):
    i, d, t = (S[k][q] / n for k in "idt"); L, C = i - d, t - d
    s = (L * C).sum() / (L * L).sum(); r = np.corrcoef(L, C)[0, 1]; big = np.abs(L) > np.percentile(np.abs(L), 90)
    print(f"{nm:16s}: slope(trained-dropped ~ intact-dropped) {s:.3f}, corr {r:.3f}; on the 10% most-shifted neurons slope {(L[big] * C[big]).sum() / (L[big] ** 2).sum():.3f}")
