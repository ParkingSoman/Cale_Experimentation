"""D5 map the trained approximation (drop block 16, block 17 trained). Interpretable numbers on the 2k validation images:
  per block-17 neuron j:  lost   = z_j(intact stream) - z_j(stream without block 16)   (the votes its block-16 parents gave)
                          change = z_j(trained block 17) - z_j(original block 17), both without block 16
  slope(change ~ lost), corr, and how |change| tracks |lost| across neurons.
  Output side: lost16 = block 16's own output + block 17's output shift; change = trained block output - original output.
usage: python d5_map.py TRAINED_NPZ"""
import os, sys
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); gelu = nn.gelu
rk = np.zeros(len(z["ana"]), int)
for c in range(1000): ix = np.where(z["yA"] == c)[0]; rk[ix] = np.arange(len(ix))
VA = np.where(rk >= 18)[0]; b16, b17 = m.blocks[16], m.blocks[17]
import copy; t17 = copy.deepcopy(b17); t17.load_weights([(k, mx.array(v)) for k, v in np.load(sys.argv[1]).items()])
Zi, Zd, Zt, Oi, Od, Ot = [], [], [], [], [], []
for a in range(0, len(VA), 100):
    x = mx.array(z["XA"][VA[a:a + 100]]).astype(mx.float32); x1 = b16(x)
    zi = b17.fc1(b17.pre(x1)); zd = b17.fc1(b17.pre(x)); zt = t17.fc1(t17.pre(x))
    Zi.append(np.array(zi).reshape(-1, 3072)); Zd.append(np.array(zd).reshape(-1, 3072)); Zt.append(np.array(zt).reshape(-1, 3072))
    Oi.append(np.array(b17(x1) - x).reshape(-1, 768)); Od.append(np.array(b17(x) - x).reshape(-1, 768)); Ot.append(np.array(t17(x) - x).reshape(-1, 768))
Zi, Zd, Zt, Oi, Od, Ot = (np.concatenate(v).astype(np.float64) for v in (Zi, Zd, Zt, Oi, Od, Ot))
lost, chg = Zi - Zd, Zt - Zd
def slope(a, b): a = a - a.mean(0); b = b - b.mean(0); return (a * b).sum(0) / ((b * b).sum(0) + 1e-12)
def corr(a, b): a = a - a.mean(0); b = b - b.mean(0); return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0) + 1e-12)
on = (Zi > 0).mean(0) > .005; s, c = slope(chg, lost), corr(chg, lost)
# weight by importance: how much each neuron's activation moves the output (write norm x activation std)
imp = np.linalg.norm(np.array(b17.fc2.weight) * np.array(b17.gamma)[:, None], axis=0) * np.array(gelu(mx.array(Zi.astype(np.float32)))).std(0)
top = np.argsort(-imp)[:300]
print(f"READ side (block-17 vote change vs the parent votes it lost), {on.sum()} alive neurons:")
print(f"  slope(change ~ lost): median {np.median(s[on]):.3f} IQR {np.percentile(s[on], 25):.3f}-{np.percentile(s[on], 75):.3f}; top-300 important: median {np.median(s[top]):.3f}")
print(f"  corr(change, lost):  median {np.median(c[on]):.3f}; top-300 {np.median(c[top]):.3f}")
print(f"  |lost| std vs |change| std across neurons: corr {np.corrcoef(lost.std(0)[on], chg.std(0)[on])[0, 1]:.3f}; mean lost std {lost.std(0)[on].mean():.3f}, change std {chg.std(0)[on].mean():.3f}")
# on-positions only (where the neuron fires in the intact model): does training restore the strong votes?
sp = [];
for j in np.where(on)[0][::5]:
    p = Zi[:, j] > 0
    if p.sum() > 50: sp.append(slope(chg[p, j:j + 1], lost[p, j:j + 1])[0])
print(f"  slope on the neuron's intact on-positions: median {np.median(sp):.3f}")
L, Cg = Oi - Od, Ot - Od; Lc, Cc = L - L.mean(0), Cg - Cg.mean(0)
print("WRITE side (block output change vs the output lost when block 16 is dropped):")
print(f"  global slope {(Lc * Cc).sum() / (Lc * Lc).sum():.3f}; corr {(Lc * Cc).sum() / np.sqrt((Lc ** 2).sum() * (Cc ** 2).sum()):.3f}; mean-shift slope {(L.mean(0) @ Cg.mean(0)) / (L.mean(0) @ L.mean(0)):.3f}")
U, sv, Vt = np.linalg.svd(Lc[::7], full_matrices=False)
for k in range(5):
    v = Vt[k]; a, b = Lc @ v, Cc @ v; print(f"  lost-output direction {k} ({sv[k] ** 2 / (sv ** 2).sum():.3f} of lost variance): slope of change {(a * b).sum() / (a * a).sum():.3f}, corr {np.corrcoef(a, b)[0, 1]:.3f}")
np.savez(os.path.join(D, "map_" + os.path.basename(sys.argv[1])), s=s, c=c, imp=imp, on=on)
