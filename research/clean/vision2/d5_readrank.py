"""D5 is training's read change low-rank (a few shared stream directions)? fc-only run: dU = U_trained - U_orig (3072 x 768).
Singular value shares; top direction vs: block 16's mean write m16, the stream mean / LN 'size' axis (top PC of the block-17
input), block-16 write PCs. Same for the write change dW."""
import os, numpy as np, mlx.core as mx
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); t = np.load(os.path.join(D, "tr_16_fc_0.0001.npz")); z = np.load(os.path.join(D, "x16.npz"))
b16, b17 = m.blocks[16], m.blocks[17]; U0 = np.array(b17.fc1.weight, np.float64); W0 = np.array(b17.fc2.weight, np.float64)
dU = t["fc1.weight"].astype(np.float64) - U0; dW = t["fc2.weight"].astype(np.float64) - W0
x = mx.array(z["XA"][:600]).astype(mx.float32); P17i = np.array(b17.pre(b16(x))).reshape(-1, 768).astype(np.float64); P17d = np.array(b17.pre(x)).reshape(-1, 768).astype(np.float64)
dmean = P17i.mean(0) - P17d.mean(0)                       # how the mean LN input of block 17 moved when block 16 was dropped
pc_d = np.linalg.svd(P17d - P17d.mean(0), full_matrices=False)[2][:3]; pc_l = np.linalg.svd((P17i - P17d) - (P17i - P17d).mean(0), full_matrices=False)[2][:3]
cos = lambda a, b: abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b)
for nm, M in (("read change dU (rows = neurons)", dU), ("write change dW^T (rows = neurons)", dW.T)):
    s = np.linalg.svd(M, compute_uv=False); sh = s ** 2 / (s ** 2).sum(); v = np.linalg.svd(M, full_matrices=False)[2][0]
    print(f"{nm}: |change|/|orig| {np.linalg.norm(M) / np.linalg.norm(U0 if 'read' in nm else W0):.3f}; top-5 singular shares {np.round(sh[:5], 3).tolist()} (rank-1 {sh[0]:.3f}, top-10 {sh[:10].sum():.3f})")
    if "read" in nm:
        print(f"   top direction vs: mean shift of block-17 LN input {cos(v, dmean):.2f}; lost-input PCs {[round(cos(v, p), 2) for p in pc_l]}; dropped-input PCs {[round(cos(v, p), 2) for p in pc_d]}")
        a = dU @ dmean / (dmean @ dmean); r = U0 @ dmean
        print(f"   per-neuron amount along the mean shift vs the neuron's own read of it (u . dmean): corr {np.corrcoef(a, r)[0, 1]:.3f}; slope {(a * r).sum() / (r * r).sum():.4f}")
