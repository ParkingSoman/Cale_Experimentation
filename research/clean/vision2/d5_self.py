"""D5 is the trained vote change a function of the neuron's OWN vote and of the LayerNorm size change? R^2 (held-out
images) of dz_j from: [z_j], [z_j, gelu(z_j)], [z_j * r] with r = sigma_intact / sigma_dropped at the position (block-17
LN scale), [r], the lost vote, and all together. 300 most important block-17 neurons."""
import os, sys, copy
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); gelu = nn.gelu
rk = np.zeros(len(z["ana"]), int)
for c in range(1000): ix = np.where(z["yA"] == c)[0]; rk[ix] = np.arange(len(ix))
VA = np.where(rk >= 18)[0][::2]; b16, b17 = m.blocks[16], m.blocks[17]; t17 = copy.deepcopy(b17); t17.load_weights([(k, mx.array(v)) for k, v in np.load(os.path.join(D, sys.argv[1])).items()])
dZ, Zo, Lost, Rr = [], [], [], []
for a in range(0, len(VA), 100):
    x = mx.array(z["XA"][VA[a:a + 100]]).astype(mx.float32); zo = b17.fc1(b17.pre(x))
    dZ.append(np.array(t17.fc1(t17.pre(x)) - zo).reshape(-1, 3072)); Zo.append(np.array(zo).reshape(-1, 3072)); Lost.append(np.array(b17.fc1(b17.pre(b16(x))) - zo).reshape(-1, 3072))
    sd = lambda v: np.array(b17.conv_dw(v)).reshape(-1, 768).std(1); Rr.append(sd(b16(x)) / sd(x))
dZ, Zo, Lost = (np.concatenate(v).astype(np.float64) for v in (dZ, Zo, Lost)); r = np.concatenate(Rr).astype(np.float64)
print(f"LN size ratio sigma_intact/sigma_dropped at block 17: median {np.median(r):.3f}, p10 {np.percentile(r, 10):.3f}, p90 {np.percentile(r, 90):.3f}")
imp = np.linalg.norm(np.array(b17.fc2.weight) * np.array(b17.gamma)[:, None], axis=0) * dZ.std(0); J = np.argsort(-imp)[:300]
h = np.repeat(np.arange(len(dZ) // 49) % 2 == 0, 49)
def r2(Y, F): A = np.c_[F, np.ones(len(F))]; w = np.linalg.lstsq(A[h], Y[h], rcond=None)[0]; return 1 - ((A[~h] @ w - Y[~h]) ** 2).mean() / Y[~h].var()
g = lambda v: np.array(gelu(mx.array(v.astype(np.float32)))).astype(np.float64)
res = {k: [] for k in ("own z", "own z, gelu", "z * r", "r", "lost", "all")}
for j in J:
    zj, dj, lj = Zo[:, j], dZ[:, j], Lost[:, j]
    for k, F in (("own z", zj[:, None]), ("own z, gelu", np.c_[zj, g(zj)]), ("z * r", (zj * r)[:, None]), ("r", r[:, None]), ("lost", lj[:, None]), ("all", np.c_[zj, g(zj), zj * r, r, lj])):
        res[k].append(r2(dj, F))
print("median held-out R^2 of the trained vote change from: " + " | ".join(f"{k} {np.median(v):.3f}" for k, v in res.items()))
