"""D5 pooled restore (target read off the trained map: training moves each block-17 neuron's per-image POOLED activation
toward its intact value; the head sees only pooled activations). For the K most important block-17 neurons, one linear
read change per neuron so that, to first order, mean_pos gelu(z_j + du_j . pre) = intact pooled activation on every image:
   g_j(img) = mean_pos[gelu'(z_j) pre],   du_j = argmin sum_img (g_j(img) . du - t_j(img))^2  (min-norm, one solve)
Target t = intact pooled - dropped pooled (the intact network's own internal values; no output loss, no step choice).
Readings: analysis ranks 0-NR-1. Score: sealed 3k. usage: python d5_pooledfix.py K [NR=6]"""
import os, sys, copy, time
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
t0 = time.time(); K = int(sys.argv[1]); NR = int(sys.argv[2]) if len(sys.argv) > 2 else 6
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); gelu = nn.gelu; b16, b17 = m.blocks[16], m.blocks[17]
rk = np.zeros(len(z["ana"]), int)
for c in range(1000): ix = np.where(z["yA"] == c)[0]; rk[ix] = np.arange(len(ix))
RD = np.where(rk < NR)[0]; W2 = np.array(b17.fc2.weight) * np.array(b17.gamma)[:, None]
def gd(v): return 0.5 * (1 + mx.erf(v / 2 ** .5)) + v * mx.exp(-v * v / 2) / (2 * np.pi) ** .5
# importance from a first pass: |write| x std of the pooled lost activation
L = []
for a in range(0, len(RD), 200):
    x = mx.array(z["XA"][RD[a:a + 200]]).astype(mx.float32); L.append(np.array(gelu(b17.fc1(b17.pre(b16(x)))).mean((1, 2)) - gelu(b17.fc1(b17.pre(x))).mean((1, 2))))
L = np.concatenate(L); imp = np.linalg.norm(W2, axis=0) * L.std(0); J = np.argsort(-imp)[:K]; Jm = mx.array(J)
A = mx.zeros((K, 768, 768)); bvec = mx.zeros((K, 768))
for a in range(0, len(RD), 50):
    x = mx.array(z["XA"][RD[a:a + 50]]).astype(mx.float32); pre = b17.pre(x).reshape(x.shape[0], 49, 768)
    zd = b17.fc1(b17.pre(x)).reshape(x.shape[0], 49, -1)[..., Jm]; t = (gelu(b17.fc1(b17.pre(b16(x)))).reshape(x.shape[0], 49, -1)[..., Jm].mean(1) - gelu(zd).mean(1))
    g = mx.einsum("bpk,bpc->bkc", gd(zd), pre) / 49                    # (B, K, 768)
    A = A + mx.einsum("bkc,bkd->kcd", g, g); bvec = bvec + mx.einsum("bkc,bk->kc", g, t); mx.eval(A, bvec)
print(f"normal equations for {K} neurons from {len(RD)} images, {time.time() - t0:.0f}s", flush=True)
A, bvec = np.array(A).astype(np.float64), np.array(bvec).astype(np.float64); dU = np.zeros((K, 768))
for i in range(K): dU[i] = np.linalg.pinv(A[i], rcond=1e-6) @ bvec[i]
U0 = np.array(b17.fc1.weight, np.float64); print(f"read change: median |du|/|u| {np.median(np.linalg.norm(dU, axis=1) / np.linalg.norm(U0[J], axis=1)):.3f}", flush=True)
def sealed(frac):
    blk = copy.deepcopy(b17); U = U0.copy(); U[J] += frac * dU; blk.fc1.weight = mx.array(U.astype(np.float32)); T = mx.array(z["LE"]); out = []
    for a in range(0, 3000, 200):
        Lg = m.head(blk(mx.array(z["XE"][a:a + 200]).astype(mx.float32))); Tt = T[a:a + 200]
        lp = Lg - mx.logsumexp(Lg, 1, keepdims=True); tp = Tt - mx.logsumexp(Tt, 1, keepdims=True); out.append(np.array((mx.exp(tp) * (tp - lp)).sum(1)))
    return float(np.concatenate(out).mean())
print(f"K {K}: sealed KL  full step (derived) {sealed(1.0):.4f} | half step {sealed(0.5):.4f}   [plain drop .1042, strong training .0942]  {time.time() - t0:.0f}s")
np.savez(os.path.join(D, f"pooledfix_K{K}.npz"), J=J, dU=dU)
