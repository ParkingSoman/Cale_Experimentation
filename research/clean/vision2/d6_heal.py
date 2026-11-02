"""D6 restore the healing for a partial cut of block 16 (law: votes must be louder by the lost vote share).
Keep k block-16 neurons (energy order, no mean). rho = median over the 300 most affected block-17 neurons of
median over their intact ON positions of (z_I - c) / (z_cut - c)  (one reading). Block-17 votes: k_gain = rho on the
normalized part (as d5_gain). Also report rho for the full drop (law: ~1). usage: python d6_heal.py"""
import os, copy, numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); b16, b17 = m.blocks[16], m.blocks[17]; gelu = nn.gelu
rk = np.zeros(len(z["ana"]), int)
for c in range(1000): ix = np.where(z["yA"] == c)[0]; rk[ix] = np.arange(len(ix))
RD = np.where(rk < 2)[0]; W1, c1 = np.array(b16.fc1.weight), np.array(b16.fc1.bias); W2 = np.array(b16.fc2.weight, np.float64) * np.array(b16.gamma, np.float64)[:, None]
b2 = (np.array(b16.fc2.bias, np.float64) * np.array(b16.gamma, np.float64)).astype(np.float32)
U = np.array(b17.fc1.weight, np.float64); bb = np.array(b17.fc1.bias, np.float64); lb = np.array(b17.norm.bias, np.float64); cc = U @ lb + bb
h2 = 0; n = 0
for a in range(0, len(RD), 100):
    h = np.array(gelu(b16.fc1(b16.pre(mx.array(z["XA"][RD[a:a + 100]]).astype(mx.float32))))).reshape(-1, 3072).astype(np.float64); h2 += (h ** 2).sum(0); n += len(h)
order = np.argsort(-(np.sqrt(h2 / n) * np.linalg.norm(W2, axis=0)))
def keepf(K):
    if len(K) == 0: return lambda x: x
    A, cA, B = mx.array(W1[K].T), mx.array(c1[K]), mx.array(W2[:, K].T.astype(np.float32)); return lambda x: x + gelu(b16.pre(x) @ A + cA) @ B + mx.array(b2)
ZI, Zi = [], []
for a in range(0, len(RD), 100):
    x = mx.array(z["XA"][RD[a:a + 100]]).astype(mx.float32); ZI.append(np.array(b17.fc1(b17.pre(b16(x)))).reshape(-1, 3072))
ZI = np.concatenate(ZI).astype(np.float64); J = None
def rho_for(f):
    global J
    Zc = np.concatenate([np.array(b17.fc1(b17.pre(f(mx.array(z["XA"][RD[a:a + 100]]).astype(mx.float32))))).reshape(-1, 3072) for a in range(0, len(RD), 100)]).astype(np.float64)
    if J is None: J = np.argsort(-np.abs(ZI - Zc).std(0) * np.linalg.norm(np.array(b17.fc2.weight), axis=0))[:300]
    rs = []
    for j in J:
        on = (ZI[:, j] > 0) & (np.abs(Zc[:, j] - cc[j]) > 1e-3)
        if on.sum() > 20: rs.append(np.median((ZI[on, j] - cc[j]) / (Zc[on, j] - cc[j])))
    return float(np.median(rs))
def sealed(f, k):
    blk = copy.deepcopy(b17); blk.fc1.weight = mx.array((k * U).astype(np.float32)); blk.fc1.bias = mx.array((bb + (1 - k) * U @ lb).astype(np.float32))
    T = mx.array(z["LE"]); out = []
    for i in range(0, 3000, 200):
        x = mx.array(z["XE"][i:i + 200]).astype(mx.float32); L = m.head(blk(f(x))); Tt = T[i:i + 200]
        lp = L - mx.logsumexp(L, 1, keepdims=True); tp = Tt - mx.logsumexp(Tt, 1, keepdims=True); out.append(np.array((mx.exp(tp) * (tp - lp)).sum(1)))
    return float(np.concatenate(out).mean())
for k in (0, 96, 384, 768, 1536):
    f = keepf(order[:k]); rho = rho_for(f)
    print(f"keep {k:4d}: rho (needed vote gain) {rho:.3f}  sealed KL no gain {sealed(f, 1.0):.4f} -> with gain rho {sealed(f, rho):.4f}", flush=True)
