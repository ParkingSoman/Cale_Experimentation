"""D5 pooled check: the head averages block 17's output over the 49 positions, so it sees per image only each neuron's
POOLED activation. Regress (trained - dropped) on (intact - dropped) of per-image pooled activations: over all
(image, neuron) pairs, per neuron, and after removing each neuron's global mean shift. Write side: pooled block output."""
import os, sys, copy
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); gelu = nn.gelu
rk = np.zeros(len(z["ana"]), int)
for c in range(1000): ix = np.where(z["yA"] == c)[0]; rk[ix] = np.arange(len(ix))
VA = np.where(rk >= 18)[0]; b16, b17 = m.blocks[16], m.blocks[17]; t17 = copy.deepcopy(b17); t17.load_weights([(k, mx.array(v)) for k, v in np.load(os.path.join(D, sys.argv[1])).items()])
P = {k: [] for k in "idt"}; O = {k: [] for k in "idt"}
for a in range(0, len(VA), 100):
    x = mx.array(z["XA"][VA[a:a + 100]]).astype(mx.float32); x1 = b16(x)
    for k, blk, xx in (("i", b17, x1), ("d", b17, x), ("t", t17, x)):
        P[k].append(np.array(gelu(blk.fc1(blk.pre(xx))).mean((1, 2)))); O[k].append(np.array((blk(xx) - x).mean((1, 2))))
P = {k: np.concatenate(v).astype(np.float64) for k, v in P.items()}; O = {k: np.concatenate(v).astype(np.float64) for k, v in O.items()}
def rep(L, C, nm):
    s = (L * C).sum() / (L * L).sum(); r = np.corrcoef(L.ravel(), C.ravel())[0, 1]
    Lc, Cc = L - L.mean(0), C - C.mean(0); s2 = (Lc * Cc).sum() / (Lc * Lc).sum(); r2 = np.corrcoef(Lc.ravel(), Cc.ravel())[0, 1]
    pn = (Lc * Cc).sum(0) / ((Lc * Lc).sum(0) + 1e-12); pc = np.array([np.corrcoef(Lc[:, j], Cc[:, j])[0, 1] if Lc[:, j].std() > 1e-9 and Cc[:, j].std() > 1e-9 else np.nan for j in range(L.shape[1])])
    print(f"{nm}: all pairs slope {s:.3f} corr {r:.3f} | per-image pattern (each neuron's mean removed) slope {s2:.3f} corr {r2:.3f} | per neuron: median slope {np.nanmedian(pn):.3f}, median corr {np.nanmedian(pc):.3f}")
rep(P["i"] - P["d"], P["t"] - P["d"], "READ  pooled activation per image x neuron")
rep(O["i"] - O["d"], O["t"] - O["d"], "WRITE pooled block output per image x channel")
