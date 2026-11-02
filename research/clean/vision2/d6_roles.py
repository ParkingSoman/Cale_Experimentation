"""D6 two roles of a block-16 neuron i (write d_i(p) = w_i h_i(p), centered across channels as LN does):
  VOLUME  vol_i  = E_p[ 2 h_i (w~_i . x~1(p)) - h_i^2 |w~_i|^2 ] / E_p[ |x~1|^2 - |x~|^2 ]   (exact share of the size
           increase block 16 makes, first order in removing i; x~1 = centered stream after block 16, x~ before)
  GATE    gate_i = sum over readers j (300 most affected) of E over j's ON positions [ h_i (u'_j . w~_i) / sigma_I ],
           normalized by the total gate the readers receive (signed; can be negative = inhibition)
Report: how volume and gate are distributed across neurons, their correlation, and where the top-96 energy neurons sit."""
import os, numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); b16, b17 = m.blocks[16], m.blocks[17]; gelu = nn.gelu
rk = np.zeros(len(z["ana"]), int)
for c in range(1000): ix = np.where(z["yA"] == c)[0]; rk[ix] = np.arange(len(ix))
RD = np.where(rk < 2)[0]; W2 = np.array(b16.fc2.weight, np.float64) * np.array(b16.gamma, np.float64)[:, None]; Wc = W2 - W2.mean(0, keepdims=True)
U = np.array(b17.fc1.weight, np.float64) * np.array(b17.norm.weight, np.float64)[None]; cc = np.array(b17.fc1.weight, np.float64) @ np.array(b17.norm.bias, np.float64) + np.array(b17.fc1.bias, np.float64)
H, X1, ZI, dS = [], [], [], 0
for a in range(0, len(RD), 100):
    x = mx.array(z["XA"][RD[a:a + 100]]).astype(mx.float32); x1 = b16(x)
    H.append(np.array(gelu(b16.fc1(b16.pre(x)))).reshape(-1, 3072)); x1n = np.array(x1).reshape(-1, 768).astype(np.float64); x0n = np.array(x).reshape(-1, 768).astype(np.float64)
    x1n -= x1n.mean(1, keepdims=True); x0n -= x0n.mean(1, keepdims=True); X1.append(x1n.astype(np.float32)); dS += ((x1n ** 2).sum(1) - (x0n ** 2).sum(1)).sum()
    ZI.append(np.array(b17.fc1(b17.pre(x1))).reshape(-1, 3072))
H, X1, ZI = np.concatenate(H).astype(np.float64), np.concatenate(X1).astype(np.float64), np.concatenate(ZI).astype(np.float64)
sig = np.sqrt((X1 ** 2).mean(1))
vol = (2 * (H * (X1 @ Wc)).sum(0) - (H ** 2).sum(0) * (Wc ** 2).sum(0)) / dS
J = np.argsort(-ZI.std(0) * np.linalg.norm(np.array(b17.fc2.weight), axis=0))[:300]; UW = U[J] @ Wc
gate = np.zeros(3072)
for q, j in enumerate(J):
    on = ZI[:, j] > 0
    if on.sum() >= 10: gate += UW[q] * (H[on] / sig[on, None]).mean(0)
gate /= np.abs(gate).sum()
en = np.sqrt((H ** 2).mean(0)) * np.linalg.norm(W2, axis=0); top = np.argsort(-en)[:96]
print(f"volume shares sum {vol.sum():.3f} (first-order); top-96 energy neurons hold volume {vol[top].sum():.3f}, gate {gate[top].sum():+.3f} (abs {np.abs(gate[top]).sum():.3f})")
for q in (96, 384, 1536):
    tv = np.argsort(-vol)[:q]; tg = np.argsort(-np.abs(gate))[:q]
    print(f"  top-{q} by volume hold volume {vol[tv].sum():.3f} and |gate| {np.abs(gate[tv]).sum():.3f} | top-{q} by |gate| hold |gate| {np.abs(gate[tg]).sum():.3f} and volume {vol[tg].sum():.3f}")
print(f"corr(volume, gate) {np.corrcoef(vol, gate)[0, 1]:.3f}; corr(volume, energy) {np.corrcoef(vol, en)[0, 1]:.3f}; corr(|gate|, energy) {np.corrcoef(np.abs(gate), en)[0, 1]:.3f}; share of gate that is negative {(-gate[gate < 0].sum()):.3f}")
np.savez(os.path.join(D, "d6_roles.npz"), vol=vol, gate=gate, en=en)
