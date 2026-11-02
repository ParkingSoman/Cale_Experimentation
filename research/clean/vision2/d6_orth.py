"""D6 how much of block 16's write is 'free' under block 17's LayerNorm? Per spot, in the READERS' metric
M = U'^T U' (U' = block-17 reads x LN weight; all vectors centered over channels as LN does, after block 17's dwconv):
  lambda = <s, d>_M / <s, s>_M ;  share of d the readers see as proportional = <s,d>_M^2 / (<s,s>_M <d,d>_M)
Also in the plain Euclidean metric. Then a direct test of the identity LN(s + d) = LN(s + d_perp / (1 + lambda)):
replace block 16 by adding only d_perp/(1+lambda) (the exact novel part, computed from the intact block at each spot)
and by adding d_perp alone; sealed KL -> how much the free part is worth."""
import os, numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); b16, b17 = m.blocks[16], m.blocks[17]
Up = np.array(b17.fc1.weight, np.float64) * np.array(b17.norm.weight, np.float64)[None]; Mm = Up.T @ Up; Mm /= np.trace(Mm) / 768
Mx = mx.array(Mm.astype(np.float32)); cen = lambda v: v - v.mean(-1, keepdims=True)
lam_r, sh_r, sh_e = [], [], []
for a in range(0, 600, 100):
    x = mx.array(z["XA"][a:a + 100]).astype(mx.float32); s = cen(b17.conv_dw(x)); d = cen(b17.conv_dw(b16(x) - x))
    sM, dM = s @ Mx, d @ Mx; sd_ = (s * dM).sum(-1); ss = (s * sM).sum(-1); dd = (d * dM).sum(-1)
    lam_r.append(np.array(sd_ / ss).ravel()); sh_r.append(np.array(sd_ ** 2 / (ss * dd)).ravel()); sh_e.append(np.array((s * d).sum(-1) ** 2 / ((s * s).sum(-1) * (d * d).sum(-1))).ravel())
lam_r, sh_r, sh_e = map(np.concatenate, (lam_r, sh_r, sh_e))
print(f"readers' metric: lambda median {np.median(lam_r):.3f} (p10 {np.percentile(lam_r, 10):.3f}, p90 {np.percentile(lam_r, 90):.3f}); proportional share of the write {np.median(sh_r):.3f}")
print(f"euclidean metric: proportional share {np.median(sh_e):.3f}")
def sealed(addf):
    T = mx.array(z["LE"]); out = []
    for i in range(0, 3000, 200):
        x = mx.array(z["XE"][i:i + 200]).astype(mx.float32); L = m.head(b17(x + addf(x))); Tt = T[i:i + 200]
        lp = L - mx.logsumexp(L, 1, keepdims=True); tp = Tt - mx.logsumexp(Tt, 1, keepdims=True); out.append(np.array((mx.exp(tp) * (tp - lp)).sum(1)))
    return float(np.concatenate(out).mean())
def perp(x, scale):
    s, d = x, b16(x) - x                                           # stream-space (before dwconv), readers' metric, centered
    sc, dc = cen(s), cen(d); lam = ((sc @ Mx) * dc).sum(-1, keepdims=True) / ((sc @ Mx) * sc).sum(-1, keepdims=True)
    dp = d - lam * sc; return dp / (1 + lam) if scale else dp
print(f"plain drop {sealed(lambda x: 0 * x):.4f} | add d_perp/(1+lambda) {sealed(lambda x: perp(x, True)):.4f} | add d_perp {sealed(lambda x: perp(x, False)):.4f} | full block {sealed(lambda x: b16(x) - x):.4f}")
