"""D1 step 1e: what ties the tiny parts of one small neuron together, in the WEIGHTS? For the top-K tiny neurons that
track a small neuron, mean pairwise |cos| of (read u * LN weight), (write gamma * fc2 column), (direct class vote =
head(write)), vs the same for random pairs from the same blocks. Also the sign pattern (do parts write the same way?).
usage: python d1_bond.py SMALL sb sn K   (needs d1_compose-style search; recomputed here)"""
import os, sys
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
X, y = load_images(); gelu = nn.gelu; m = load_convnext(); ST = m.stage_of
CFG = {"atto": ((2, 2, 6, 2), (40, 80, 160, 320)), "nano": ((2, 2, 8, 2), (80, 160, 320, 640))}
small, sb, sn, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); sm = load_convnext(small, *CFG[small])
stage = sm.stage_of[sb]; ana, _ = split(y); I = ana[::20]; rng = np.random.default_rng(0)
blocks = [b for b in range(len(m.blocks)) if ST[b] == stage]; T, S = [], []
for a in range(0, len(I), 50):
    xs = mx.array(norm_img(np.asarray(X[I[a:a + 50]]))); x = m.stem_norm(m.stem(xs)); acc = []
    for b in range(blocks[-1] + 1):
        if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
        if ST[b] == stage: acc.append(gelu(m.blocks[b].fc1(m.blocks[b].pre(x))))
        x = m.blocks[b](x)
    h = mx.concatenate(acc, -1); B, R, C, Kk = h.shape; pick = np.stack([rng.choice(R * C, 30, replace=False) for _ in range(B)])
    idx = mx.array((np.arange(B)[:, None] * R * C + pick).reshape(-1)); T.append(np.array(h.reshape(-1, Kk)[idx]))
    x = sm.stem_norm(sm.stem(xs)); S2 = sm.stage_of
    for b in range(sb + 1):
        if b > 0 and S2[b] != S2[b - 1]: x = sm.down[S2[b] - 1](sm.down_norm[S2[b] - 1](x))
        if b == sb: S.append(np.array(gelu(sm.blocks[b].fc1(sm.blocks[b].pre(x))[..., sn]).reshape(-1)[idx]))
        x = sm.blocks[b](x)
T = np.concatenate(T).astype(np.float64); s = np.concatenate(S).astype(np.float64)
ids = [(b, n) for b in blocks for n in range(m.blocks[b].fc1.weight.shape[0])]
c = ((T - T.mean(0)) / (T.std(0) + 1e-9) * ((s - s.mean()) / s.std())[:, None]).mean(0); top = [ids[i] for i in np.argsort(-c)[:K]]
def vecs(n):
    b, j = n; blk = m.blocks[b]; u = np.array(blk.fc1.weight[j]) * np.array(blk.norm.weight); w = np.array(blk.fc2.weight[:, j]) * np.array(blk.gamma)
    return u, w
# direct class vote: write -> (later downsample ignored inside stage 4 only) -> head norm weight -> fc; stage 4 only
Wh = np.array(m.head_fc.weight) if hasattr(m, "head_fc") else None
def stats(nodes):
    U = np.stack([vecs(n)[0] for n in nodes]); Wr = np.stack([vecs(n)[1] for n in nodes])
    nu = U / np.linalg.norm(U, axis=1, keepdims=True); nw = Wr / np.linalg.norm(Wr, axis=1, keepdims=True)
    iu = np.triu_indices(len(nodes), 1); cu, cw = (nu @ nu.T)[iu], (nw @ nw.T)[iu]
    return np.abs(cu).mean(), cu.mean(), np.abs(cw).mean(), cw.mean()
rand = [ids[i] for i in rng.choice(len(ids), K, replace=False)]
print(f"{small} {sb}/{sn}: top {K} tiny parts {top[:8]} ...")
for name, nodes in (("parts", top), ("random", rand)):
    a, b_, c_, d_ = stats(nodes); print(f"  {name:7s} read |cos| {a:.3f} (signed {b_:+.3f})   write |cos| {c_:.3f} (signed {d_:+.3f})")
# shared components: top singular direction of parts' reads / writes, and its share of each part's vector
U = np.stack([vecs(n)[0] / np.linalg.norm(vecs(n)[0]) for n in top]); Wr = np.stack([vecs(n)[1] / np.linalg.norm(vecs(n)[1]) for n in top])
for name, M in (("read", U), ("write", Wr)):
    sv = np.linalg.svd(M, compute_uv=False); print(f"  {name} singular values^2 share (top 5): " + " ".join(f"{v:.2f}" for v in (sv ** 2 / (sv ** 2).sum())[:5]))
np.save(os.path.join(HF, "imagenet", "d1", f"parts_{small}_{sb}_{sn}.npy"), np.array([ids[i] for i in np.argsort(-c)[:100]]))
np.save(os.path.join(HF, "imagenet", "d1", f"partcorr_{small}_{sb}_{sn}.npy"), np.sort(c)[::-1][:100])
