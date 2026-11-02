"""D6 law-based cut order for block 16. Per block-16 neuron i (write d_i(p) = w_i h_i(p)):
  size share  eta_i = E_p[|d_i(p)|^2] / sum_k E_p[|d_k(p)|^2]   (its contribution to the stream size it adds)
  vote share  v_i   = sum over the 300 most affected block-17 readers j, on j's ON positions, of E[|u_j' . d_i| / sigma]
                      normalized to sum 1 (u_j' = u_j * LN17 weight; dwconv ignored)
Self-healing when cut: vote share ~ size share. Orders (keep k, cut the rest, no mean): energy (keep large), reverse
energy (keep small), ratio (keep large v/eta: they carry votes the size drop cannot heal). usage: python d6_ratio.py"""
import os, numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); b16, b17 = m.blocks[16], m.blocks[17]; gelu = nn.gelu
rk = np.zeros(len(z["ana"]), int)
for c in range(1000): ix = np.where(z["yA"] == c)[0]; rk[ix] = np.arange(len(ix))
RD = np.where(rk < 2)[0]; W1, c1 = np.array(b16.fc1.weight), np.array(b16.fc1.bias); W2 = np.array(b16.fc2.weight, np.float64) * np.array(b16.gamma, np.float64)[:, None]
b2 = (np.array(b16.fc2.bias, np.float64) * np.array(b16.gamma, np.float64)).astype(np.float32)
U = np.array(b17.fc1.weight, np.float64) * np.array(b17.norm.weight, np.float64)[None]
H, ZI, SG = [], [], []
for a in range(0, len(RD), 100):
    x = mx.array(z["XA"][RD[a:a + 100]]).astype(mx.float32); x1 = b16(x)
    H.append(np.array(gelu(b16.fc1(b16.pre(x)))).reshape(-1, 3072)); ZI.append(np.array(b17.fc1(b17.pre(x1))).reshape(-1, 3072))
    SG.append(np.sqrt(np.array(b17.conv_dw(x1)).reshape(-1, 768).var(1) + 1e-6))
H, ZI = np.concatenate(H).astype(np.float64), np.concatenate(ZI).astype(np.float64); sg = np.concatenate(SG).astype(np.float64)
wn2 = (W2 ** 2).sum(0); eta = (H ** 2).mean(0) * wn2; eta /= eta.sum(); en = np.sqrt(eta)
J = np.argsort(-ZI.std(0) * np.linalg.norm(np.array(b17.fc2.weight), axis=0))[:300]
Wc = W2 - W2.mean(0, keepdims=True); UW = U[J] @ Wc            # (300, 3072): reader j's read of neuron i's (centered) write
vs = np.zeros(3072)
for q, j in enumerate(J):
    on = ZI[:, j] > 0
    if on.sum() < 10: continue
    vs += np.abs(UW[q])[None, :].repeat(1, 0)[0] * (H[on] / sg[on, None]).mean(0)
vs /= vs.sum(); ratio = vs / (eta + 1e-12)
print(f"vote share vs size share: corr(log) {np.corrcoef(np.log(vs + 1e-9), np.log(eta + 1e-9))[0, 1]:.3f}; ratio median {np.median(ratio):.2f}, p10 {np.percentile(ratio, 10):.2f}, p90 {np.percentile(ratio, 90):.2f}")
def keepf(K):
    if len(K) == 0: return lambda x: x
    A, cA, B = mx.array(W1[K].T), mx.array(c1[K]), mx.array(W2[:, K].T.astype(np.float32)); return lambda x: x + gelu(b16.pre(x) @ A + cA) @ B + mx.array(b2)
def sealed(f):
    T = mx.array(z["LE"]); out = []
    for i in range(0, 3000, 200):
        x = mx.array(z["XE"][i:i + 200]).astype(mx.float32); L = m.head(b17(f(x))); Tt = T[i:i + 200]
        lp = L - mx.logsumexp(L, 1, keepdims=True); tp = Tt - mx.logsumexp(Tt, 1, keepdims=True); out.append(np.array((mx.exp(tp) * (tp - lp)).sum(1)))
    return float(np.concatenate(out).mean())
print("k kept | energy (keep large) | reverse energy (keep small) | law ratio (keep high vote/size)")
for k in (96, 384, 768, 1536, 2304):
    print(f"{k:5d} | {sealed(keepf(np.argsort(-en)[:k])):.4f} | {sealed(keepf(np.argsort(en)[:k])):.4f} | {sealed(keepf(np.argsort(-ratio)[:k])):.4f}", flush=True)
print("drop all (k=0) .1042; intact (k=3072) 0")
np.savez(os.path.join(D, "d6_ratio.npz"), eta=eta, vs=vs)
