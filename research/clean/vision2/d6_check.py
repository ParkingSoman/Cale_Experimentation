"""Checks: (1) lambda and proportional share in the readers' metric with per-spot FLUCTUATIONS only (subtract the mean of s
and of d over all spots and images before projecting); (2) within-stage-3 corr of novelty/r with drop KL (law_all)."""
import os, numpy as np, mlx.core as mx
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); b16, b17 = m.blocks[16], m.blocks[17]
Up = np.array(b17.fc1.weight, np.float64) * np.array(b17.norm.weight, np.float64)[None]; Mm = Up.T @ Up; Mm /= np.trace(Mm) / 768
x = mx.array(z["XA"][:600]).astype(mx.float32); cen = lambda v: v - v.mean(-1, keepdims=True)
S = np.array(cen(b17.conv_dw(x))).reshape(-1, 768).astype(np.float64); Dd = np.array(cen(b17.conv_dw(b16(x) - x))).reshape(-1, 768).astype(np.float64)
for nm, s, d in (("raw", S, Dd), ("fluctuations only", S - S.mean(0), Dd - Dd.mean(0))):
    sM, dM = s @ Mm, d @ Mm; sd_, ss, dd = (s * dM).sum(1), (s * sM).sum(1), (d * dM).sum(1)
    print(f"{nm:18s}: lambda median {np.median(sd_ / ss):.3f}; proportional share median {np.median(sd_ ** 2 / (ss * dd)):.3f}; pooled share {sd_.sum() ** 2 / (ss.sum() * dd.sum()):.3f}")
mS, mD = S.mean(0), Dd.mean(0); print(f"mean-direction share of |s|_M^2 {(mS @ Mm @ mS) / np.mean((S * (S @ Mm)).sum(1)):.3f}, of |d|_M^2 {(mD @ Mm @ mD) / np.mean((Dd * (Dd @ Mm)).sum(1)):.3f}; cos_M(mean s, mean d) {(mS @ Mm @ mD) / np.sqrt((mS @ Mm @ mS) * (mD @ Mm @ mD)):.3f}")
A = np.load(os.path.join(D, "law_all.npy")); st3 = A[:, 0] <= 13
print(f"within stage 3 (8 blocks): corr(novelty/r, drop KL) {np.corrcoef(A[st3, 6], A[st3, 7])[0, 1]:.3f}; all 10: {np.corrcoef(A[:, 6], A[:, 7])[0, 1]:.3f}")
