"""E1: direct-path edges for the whole network. Rules: VISION-LOG.md E1."""
import json, time
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *

NB = 18; PPI = 49
m = load_convnext(); X, y = load_images(); ana, _ = split(y)
IDX = np.concatenate([ana[y[ana] == c][:3] for c in range(1000)])
H = [m.blocks[b].fc1.weight.shape[0] for b in range(NB)]; OFF = np.concatenate([[0], np.cumsum(H)]); ST = m.stage_of
rng = np.random.default_rng(0)


def maps(xb, want_sigma_down=False):
    x = m.stem_norm(m.stem(mx.array(norm_img(xb)))); Hh, SIG, sd = [], [], []
    for b in range(NB):
        if b > 0 and ST[b] != ST[b - 1]:
            sd.append(mx.mean(1 / mx.sqrt(mx.var(x, axis=-1) + 1e-6)))
            x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
        blk = m.blocks[b]; d = blk.conv_dw(x); SIG.append(1 / mx.sqrt(mx.var(d, axis=-1, keepdims=True) + 1e-6))
        Hh.append(nn.gelu(blk.fc1(blk.norm(d)))); x = blk(x)
    return Hh, SIG, sd


# mean 1/sigma at each downsample LayerNorm (200 images)
_, _, sd = maps(np.asarray(X[IDX[:600:3]])); SD = [float(v) for v in sd]
DSUM = [np.array(m.down[t].weight).sum((1, 2)) for t in range(3)]          # (out, in)
DNW = [np.array(m.down_norm[t].weight) for t in range(3)]


def to_stage(u, s0, s1):
    for t in range(s0, s1):
        u = DSUM[t] @ (DNW[t][:, None] * (u - u.mean(0, keepdims=True)) * SD[t])
    return u


U = [np.array(m.blocks[b].fc2.weight) * np.array(m.blocks[b].gamma)[:, None] for b in range(NB)]   # (C, H) writes
V = {}
for b in range(1, NB):
    blk = m.blocks[b]; ks = np.array(blk.conv_dw.weight).sum((1, 2, 3)); R = np.array(blk.fc1.weight) * np.array(blk.norm.weight)[None]
    W = np.concatenate([to_stage(U[p], ST[p], ST[b]) for p in range(b)], 1); W = ks[:, None] * W; W = W - W.mean(0, keepdims=True)
    V[b] = mx.array((R @ W).astype(np.float32))

t0 = time.time(); A = {b: mx.zeros((H[b], int(OFF[b]))) for b in range(1, NB)}; SW = {b: mx.zeros((H[b],)) for b in range(1, NB)}
for i in range(0, len(IDX), 25):
    Hh, SIG, _ = maps(np.asarray(X[IDX[i:i + 25]])); B = Hh[0].shape[0]
    for b in range(1, NB):
        _, R_, C_, _ = Hh[b].shape; HW = R_ * C_
        sel = mx.array(np.concatenate([j * HW + (np.arange(HW) if PPI >= HW else rng.choice(HW, PPI, replace=False)) for j in range(B)]))
        hc = Hh[b].reshape(-1, H[b])[sel]; w = mx.maximum(hc, 0)       # GELU output ~ relu(z) on the on-region
        s = SIG[b].reshape(-1, 1)[sel]
        par = mx.concatenate([(nn.AvgPool2d(2 ** (ST[b] - ST[p]), 2 ** (ST[b] - ST[p]))(Hh[p]) if ST[b] > ST[p] else Hh[p]).reshape(-1, H[p])[sel] for p in range(b)], 1)
        A[b] += w.T @ (par * s); SW[b] += w.sum(0); mx.eval(A[b], SW[b])
    if i % 500 == 0: print(i, round(time.time() - t0), "s", flush=True)

E = {}
for b in range(1, NB):
    E[b] = np.array(V[b] * A[b] / mx.maximum(SW[b], 1e-9)[:, None]).astype(np.float32)
np.savez(os.path.join(HF, "imagenet", "e1_edges.npz"), **{str(b): E[b] for b in E})
print("edges saved", round(time.time() - t0), "s")

