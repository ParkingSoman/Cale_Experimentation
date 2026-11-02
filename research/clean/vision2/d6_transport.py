"""D6 distribution-preserving fix after dropping block 16: per block-17 neuron, map the dropped vote's distribution onto
the intact vote's distribution (readings, analysis ranks 0-5; no output loss):
  moment matching : z' = mu_I + (sd_I / sd_D)(z_D - mu_D)     (affine, folds into fc1)
  quantile map    : z' = F_I^-1(F_D(z_D))                      (monotone 1-D transport, 64 quantile knots, piecewise linear)
  upper-tail moment matching: same as moment matching but on spots where z is in its top 5% (dropped / intact quantiles)
Score sealed KL. [plain drop .1042, best training .0836]"""
import os, copy, numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); b16, b17 = m.blocks[16], m.blocks[17]; gelu = nn.gelu
rk = np.zeros(len(z["ana"]), int)
for c in range(1000): ix = np.where(z["yA"] == c)[0]; rk[ix] = np.arange(len(ix))
RD = np.where(rk < 6)[0]; rng = np.random.default_rng(0); ZI, ZD = [], []
for a in range(0, len(RD), 100):
    x = mx.array(z["XA"][RD[a:a + 100]]).astype(mx.float32); k = rng.choice(x.shape[0] * 49, 600, replace=False)
    ZI.append(np.array(b17.fc1(b17.pre(b16(x)))).reshape(-1, 3072)[k]); ZD.append(np.array(b17.fc1(b17.pre(x))).reshape(-1, 3072)[k])
ZI, ZD = np.concatenate(ZI).astype(np.float64), np.concatenate(ZD).astype(np.float64)
qs = np.r_[np.linspace(0, .9, 24, endpoint=False), np.linspace(.9, 1, 41)]; QI, QD = np.quantile(ZI, qs, axis=0), np.quantile(ZD, qs, axis=0)  # (65, 3072)
U0, c0 = np.array(b17.fc1.weight, np.float64), np.array(b17.fc1.bias, np.float64)
def run(remap):
    T = mx.array(z["LE"]); out = []
    for i in range(0, 3000, 200):
        x = mx.array(z["XE"][i:i + 200]).astype(mx.float32); zz = b17.fc1(b17.pre(x)); zz = remap(zz)
        h = x + b17.gamma * b17.fc2(gelu(zz)); L = m.head(h); Tt = T[i:i + 200]
        lp = L - mx.logsumexp(L, 1, keepdims=True); tp = Tt - mx.logsumexp(Tt, 1, keepdims=True); out.append(np.array((mx.exp(tp) * (tp - lp)).sum(1)))
    return float(np.concatenate(out).mean())
mu_I, sd_I, mu_D, sd_D = ZI.mean(0), ZI.std(0), ZD.mean(0), ZD.std(0); bm = sd_I / sd_D
print(f"spread ratio sd_I/sd_D: median {np.median(bm):.3f}; mean shift mu_I - mu_D median {np.median(mu_I - mu_D):+.3f}")
A_, B_ = mx.array((mu_I - bm * mu_D).astype(np.float32)), mx.array(bm.astype(np.float32))
print(f"identity (plain drop): {run(lambda v: v):.4f}")
print(f"moment matching (all spots): {run(lambda v: A_ + B_ * v):.4f}", flush=True)
t = ZD > np.quantile(ZD, .95, axis=0); ti = ZI > np.quantile(ZI, .95, axis=0)
mt = lambda Zs, s: (np.nansum(np.where(s, Zs, np.nan), 0) / s.sum(0), np.sqrt(np.nanvar(np.where(s, Zs, np.nan), 0)))
(mI, sI), (mD, sD) = mt(ZI, ti), mt(ZD, t); bt = sI / np.maximum(sD, 1e-6)
At, Bt = mx.array((mI - bt * mD).astype(np.float32)), mx.array(bt.astype(np.float32)); print(f"upper-tail moment matching: {run(lambda v: At + Bt * v):.4f}  (tail spread ratio median {np.median(bt):.3f})", flush=True)
QDm, QIm = mx.array(QD.astype(np.float32)), mx.array(QI.astype(np.float32)); K = QD.shape[0]
def qmap(v):
    sh = v.shape; v2 = v.reshape(-1, 3072); idx = mx.clip(mx.sum(v2[:, None, :] > QDm[None], axis=1) - 1, 0, K - 2)
    lo = mx.take_along_axis(QDm, idx, 0) if False else None
    ar = mx.arange(3072)[None, :]; d0, d1 = QDm[idx, ar], QDm[idx + 1, ar]; i0, i1 = QIm[idx, ar], QIm[idx + 1, ar]
    w = mx.clip((v2 - d0) / mx.maximum(d1 - d0, 1e-6), 0, 1); return (i0 + w * (i1 - i0)).reshape(sh)
print(f"quantile map (1-D transport per neuron): {run(qmap):.4f}")
