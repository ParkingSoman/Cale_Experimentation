"""D6 is the self-healing law general? For each block b with a same-stage next block (6-13, 15, 16), with block b removed:
  r = sigma_D/sigma_I at block b+1's LN; on the ON spots of block b+1's 300 most affected neurons: lost gate G, LN
  healing (1-r)(z_D-c), healed share; slope of G on (1-r)(z_I-c) (law: 1); mean |novelty| / r (predicted vote damage).
Actual damage: sealed KL of the whole model with block b skipped (1,000 sealed images). Then: does the law's novelty
predict which blocks can be dropped (corr across blocks)? 500 analysis images for readings."""
import os, numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
m = load_convnext(); X, y = load_images(); ST = m.stage_of; gelu = nn.gelu
rank = np.zeros(len(y), int)
for c in range(1000): ix = np.where(y == c)[0]; rank[ix] = np.arange(len(ix))
AN = np.where(rank == 0)[0][::2]; EV = np.where(rank == 20)[0]
def stream_to(xs, b):
    x = m.stem_norm(m.stem(xs))
    for i in range(b):
        if i > 0 and ST[i] != ST[i - 1]: x = m.down[ST[i] - 1](m.down_norm[ST[i] - 1](x))
        x = m.blocks[i](x)
    if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
    return x
def full(xs, skip=None):
    x = m.stem_norm(m.stem(xs))
    for i in range(18):
        if i > 0 and ST[i] != ST[i - 1]: x = m.down[ST[i] - 1](m.down_norm[ST[i] - 1](x))
        if i != skip: x = m.blocks[i](x)
    return m.head(x)
rows = []
for b in [6, 7, 8, 9, 10, 11, 12, 13, 15, 16]:
    B0, B1 = m.blocks[b], m.blocks[b + 1]; C = B1.fc1.weight.shape[1]; U = np.array(B1.fc1.weight, np.float64)
    cc = U @ np.array(B1.norm.bias, np.float64) + np.array(B1.fc1.bias, np.float64)
    ZI, ZD, R = [], [], []
    for a in range(0, len(AN), 50):
        x = stream_to(mx.array(norm_img(np.asarray(X[AN[a:a + 50]]))), b); x1 = B0(x)
        ZI.append(np.array(B1.fc1(B1.pre(x1))).reshape(-1, U.shape[0])); ZD.append(np.array(B1.fc1(B1.pre(x))).reshape(-1, U.shape[0]))
        sd = lambda v: np.sqrt(np.array(B1.conv_dw(v)).reshape(-1, C).var(1) + 1e-6); R.append(sd(x) / sd(x1))
    ZI, ZD = np.concatenate(ZI).astype(np.float64), np.concatenate(ZD).astype(np.float64); r = np.concatenate(R).astype(np.float64)[:, None]
    G = ZI - cc - r * (ZD - cc); P = (1 - r) * (ZI - cc); Nv = G - P
    J = np.argsort(-np.abs(ZI - ZD).std(0) * np.linalg.norm(np.array(B1.fc2.weight), axis=0))[:300]
    gs, hs, sl, nov = [], [], [], []
    for j in J:
        on = ZI[:, j] > 0
        if on.sum() < 20: continue
        g, h = G[on, j].mean(), ((1 - r[on, 0]) * (ZD[on, j] - cc[j])).mean(); gs.append(g); hs.append(h)
        pc = P[:, j] - P[:, j].mean(); sl.append(((G[:, j] - G[:, j].mean()) @ pc) / (pc @ pc + 1e-12)); nov.append((np.abs(Nv[:, j]) / r[:, 0]).mean())
    kl = []
    for a in range(0, len(EV), 50):
        xs = mx.array(norm_img(np.asarray(X[EV[a:a + 50]]))); L = full(xs, b); T = full(xs)
        lp = L - mx.logsumexp(L, 1, keepdims=True); tp = T - mx.logsumexp(T, 1, keepdims=True); kl.append(np.array((mx.exp(tp) * (tp - lp)).sum(1)))
    row = (b, float(np.median(r)), float(np.median(gs)), float(np.median(hs)), float(np.median(np.array(hs) / np.array(gs))), float(np.median(sl)), float(np.median(nov)), float(np.concatenate(kl).mean()))
    rows.append(row); print("block %2d: r %.3f | ON: gate %+.2f healing %+.2f healed share %.2f | law slope %.3f | novelty/r %.3f | drop KL %.4f" % row, flush=True)
A = np.array(rows)
print(f"across blocks: corr(novelty/r, drop KL) {np.corrcoef(A[:, 6], A[:, 7])[0, 1]:.3f}; corr(1 - healed share, drop KL) {np.corrcoef(1 - A[:, 4], A[:, 7])[0, 1]:.3f}; corr(r, drop KL) {np.corrcoef(A[:, 1], A[:, 7])[0, 1]:.3f}")
np.save(os.path.join(HF, "imagenet", "d5cache", "law_all.npy"), A)
