"""D6 exact split for a PARTIAL cut of block 16 (keep k by energy) vs the full drop. With z_C the vote after the cut:
   z_I = c + r_C (z_C - c) + G_C,   r_C = sigma_C / sigma_I,   G_C = gate lost with the cut neurons.
For the 300 most affected block-17 neurons: on intact ON positions and on WRONG positions (fire only after the cut),
median G_C, native excess (1 - r_C)(z_C - c), net lost vote; r_C; and the counts of ON / WRONG positions.
Also: does the cut lose ON firing (missed) or add WRONG firing (false), weighted by the reader's write size?"""
import os, numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); b16, b17 = m.blocks[16], m.blocks[17]; gelu = nn.gelu
rk = np.zeros(len(z["ana"]), int)
for c in range(1000): ix = np.where(z["yA"] == c)[0]; rk[ix] = np.arange(len(ix))
VA = np.where(rk >= 18)[0][::2]; W1, c1 = np.array(b16.fc1.weight), np.array(b16.fc1.bias); W2 = np.array(b16.fc2.weight, np.float64) * np.array(b16.gamma, np.float64)[:, None]
b2 = (np.array(b16.fc2.bias, np.float64) * np.array(b16.gamma, np.float64)).astype(np.float32)
U = np.array(b17.fc1.weight, np.float64); cc = U @ np.array(b17.norm.bias, np.float64) + np.array(b17.fc1.bias, np.float64); W17 = np.linalg.norm(np.array(b17.fc2.weight), axis=0)
d = np.load(os.path.join(D, "d6_ratio.npz")); order = np.argsort(-d["eta"])
def keepf(K):
    if len(K) == 0: return lambda x: x
    A, cA, B = mx.array(W1[K].T), mx.array(c1[K]), mx.array(W2[:, K].T.astype(np.float32)); return lambda x: x + gelu(b16.pre(x) @ A + cA) @ B + mx.array(b2)
sd = lambda v: np.sqrt(np.array(b17.conv_dw(v)).reshape(-1, 768).var(1) + 1e-6)
ZI = []; SI = []; XS = []
for a in range(0, len(VA), 100):
    x = mx.array(z["XA"][VA[a:a + 100]]).astype(mx.float32); x1 = b16(x); ZI.append(np.array(b17.fc1(b17.pre(x1))).reshape(-1, 3072)); SI.append(sd(x1))
ZI = np.concatenate(ZI).astype(np.float64); SI = np.concatenate(SI).astype(np.float64)
J = None
for k in (0, 96, 384, 1536):
    f = keepf(order[:k]); ZC, SC = [], []
    for a in range(0, len(VA), 100):
        xc = f(mx.array(z["XA"][VA[a:a + 100]]).astype(mx.float32)); ZC.append(np.array(b17.fc1(b17.pre(xc))).reshape(-1, 3072)); SC.append(sd(xc))
    ZC = np.concatenate(ZC).astype(np.float64); r = (np.concatenate(SC).astype(np.float64) / SI)[:, None]
    G = ZI - cc - r * (ZC - cc); nat = (1 - r) * (ZC - cc)
    if J is None: J = np.argsort(-W17 * np.abs(ZI - ZC).std(0))[:300]
    rows = []; miss = false = 0
    for j in J:
        on, wr = ZI[:, j] > 0, (ZC[:, j] > 0) & (ZI[:, j] <= 0)
        hI, hC = np.array(gelu(mx.array(ZI[:, j].astype(np.float32)))), np.array(gelu(mx.array(ZC[:, j].astype(np.float32))))
        miss += W17[j] * np.clip(hI - hC, 0, None)[on].sum(); false += W17[j] * np.clip(hC - hI, 0, None)[~on].sum()
        rows.append((np.median(G[on, j]) if on.sum() else np.nan, np.median(nat[on, j]) if on.sum() else np.nan, np.median((ZC - ZI)[on, j]) if on.sum() else np.nan,
                     np.median(G[wr, j]) if wr.sum() > 5 else np.nan, np.median(nat[wr, j]) if wr.sum() > 5 else np.nan, on.sum(), wr.sum()))
    A = np.array(rows); md = lambda q: np.nanmedian(A[:, q])
    print(f"keep {k:4d}: r_C median {np.median(r):.3f} | ON: lost gate {md(0):+.2f} native excess {md(1):+.2f} net vote change {md(2):+.2f} | WRONG: gate {md(3):+.2f} native {md(4):+.2f}, wrong/ON positions {A[:, 6].sum() / A[:, 5].sum():.3f} | missed firing {miss:.0f} vs false firing {false:.0f}", flush=True)
