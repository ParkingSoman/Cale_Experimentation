"""D6 exact decomposition of a block-17 vote when block 16 is removed:
   z_I = c + r (z_D - c) + G,   c = u.lnb + b,  r = sigma_D / sigma_I (per position),  G = block 16's gate input.
Check exactness, then on each neuron's intact ON positions and on its "wrong" positions (fires only when dropped):
   lost vote z_D - z_I = (1 - r)(z_D - c) - G  ->  how much does the louder native vote cancel the missing gate?"""
import os, numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); b16, b17 = m.blocks[16], m.blocks[17]
rk = np.zeros(len(z["ana"]), int)
for c in range(1000): ix = np.where(z["yA"] == c)[0]; rk[ix] = np.arange(len(ix))
VA = np.where(rk >= 18)[0]; U = np.array(b17.fc1.weight, np.float64); cc = U @ np.array(b17.norm.bias, np.float64) + np.array(b17.fc1.bias, np.float64)
ZI, ZD, R = [], [], []
for a in range(0, len(VA), 100):
    x = mx.array(z["XA"][VA[a:a + 100]]).astype(mx.float32); x1 = b16(x)
    ZI.append(np.array(b17.fc1(b17.pre(x1))).reshape(-1, 3072)); ZD.append(np.array(b17.fc1(b17.pre(x))).reshape(-1, 3072))
    sd = lambda v: np.sqrt(np.array(b17.conv_dw(v)).reshape(-1, 768).var(1) + 1e-6); R.append(sd(x) / sd(x1))
ZI, ZD = np.concatenate(ZI).astype(np.float64), np.concatenate(ZD).astype(np.float64); r = np.concatenate(R).astype(np.float64)[:, None]
G = ZI - cc - r * (ZD - cc); nat = (1 - r) * (ZD - cc)          # G is exact by definition; nat = the native excess
W2 = np.array(b17.fc2.weight) * np.array(b17.gamma)[:, None]; imp = np.linalg.norm(W2, axis=0) * np.abs(ZI - ZD).std(0); J = np.argsort(-imp)[:300]
print(f"r = sigma_D/sigma_I: median {np.median(r):.3f}")
rows = []
for j in J:
    on = ZI[:, j] > 0; wrong = (ZD[:, j] > 0) & (ZI[:, j] <= 0); lost = ZD[:, j] - ZI[:, j]
    if on.sum() < 30: continue
    rows.append((G[on, j].mean(), nat[on, j].mean(), lost[on].mean(), G[wrong, j].mean() if wrong.sum() > 30 else np.nan, nat[wrong, j].mean() if wrong.sum() > 30 else np.nan,
                 np.corrcoef(G[:, j], nat[:, j])[0, 1], on.mean(), wrong.mean(), np.corrcoef(G[:, j], ZI[:, j])[0, 1]))
A = np.array(rows); md = lambda k: np.nanmedian(A[:, k])
print(f"300 most affected block-17 neurons (medians):")
print(f"  on intact ON positions:  gate input G {md(0):+.2f} | native excess (1-r)(z_D-c) {md(1):+.2f} | net lost vote {md(2):+.2f}")
print(f"  on WRONG positions (fire only when dropped): G {md(3):+.2f} | native excess {md(4):+.2f}")
print(f"  corr(G, native excess) over positions {md(5):+.3f};  corr(G, intact vote) {md(8):+.3f}")
print(f"  share of positions: intact ON {md(6):.4f}, wrong {md(7):.4f};  share of neurons whose G is positive on ON positions {np.mean(A[:, 0] > 0):.2f}")
print(f"  native excess cancels the missing gate on ON positions (ratio native/G): median {np.nanmedian(A[:, 1] / A[:, 0]):.2f}")
