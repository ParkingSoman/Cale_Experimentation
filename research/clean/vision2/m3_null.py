"""M3: positive- and negative-pole parents, view and point, with nulls and threshold curve. Rules: VISION-LOG.md M3."""
import json, time
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *

NB = 18; NULLN = 256; PPI = 49
m = load_convnext(); X, y = load_images(); ana, sea = split(y)
IDX = np.concatenate([ana[y[ana] == c][:3] for c in range(1000)])
H = [m.blocks[b].fc1.weight.shape[0] for b in range(NB)]; OFF = np.concatenate([[0], np.cumsum(H)])
rng = np.random.default_rng(0)


def maps(xb):
    x = m.stem_norm(m.stem(mx.array(norm_img(xb)))); Z, PRE = [], []
    for b in range(NB):
        if b > 0 and m.stage_of[b] != m.stage_of[b - 1]: x = m.down[m.stage_of[b] - 1](m.down_norm[m.stage_of[b] - 1](x))
        pre = m.blocks[b].pre(x); Z.append(m.blocks[b].fc1(pre)); PRE.append(pre); x = m.blocks[b](x)
    return Z, PRE


def to_res(z, pb, cb, view, sign):
    k = 2 ** (m.stage_of[cb] - m.stage_of[pb]); z = z * sign          # min-pool = -maxpool(-z)
    if k > 1: z = nn.MaxPool2d(k, k)(z)
    if view: z = nn.MaxPool2d(7, 1, 3)(z)
    return z * sign


# pass 1: per-neuron most negative value (200 images); null readouts as M2
Zs, Ps = maps(np.asarray(X[IDX[:600:3]]))
TAU = [mx.array(.5 * np.array(z.min((0, 1, 2)))) for z in Zs]
NULL = {}
for b in range(1, NB):
    W = np.array(m.blocks[b].fc1.weight); src = rng.choice(H[b], NULLN, replace=False)
    Wn = np.stack([rng.permutation(W[s]) for s in src]); pre = np.array(Ps[b].reshape(-1, Ps[b].shape[-1])); zr = np.array(Zs[b].reshape(-1, H[b]))
    raw = pre @ Wn.T; rate = (zr[:, src] > 0).mean(0)
    NULL[b] = (mx.array(Wn.T), mx.array(-np.array([np.quantile(raw[:, j], 1 - max(rate[j], 1e-4)) for j in range(NULLN)]).astype(np.float32)))
del Zs, Ps

t0 = time.time(); KEYS = [(v, s) for v in ("view", "point") for s in ("pos", "neg")]
acc = {b: {**{("n",) + k: mx.zeros((NULLN, int(OFF[b]))) for k in KEYS},
           **{("br",) + k: mx.zeros((int(OFF[b]),)) for k in KEYS}, "won": mx.zeros((H[b],)), "on": mx.zeros((H[b],)),
           "nwon": mx.zeros((NULLN,)), "non": mx.zeros((NULLN,)), "rows": 0} for b in range(1, NB)}
for i in range(0, len(IDX), 25):
    Z, PRE = maps(np.asarray(X[IDX[i:i + 25]])); B = Z[0].shape[0]
    for b in range(1, NB):
        _, Hh, Ww, _ = Z[b].shape; HW = Hh * Ww
        sel = mx.array(np.concatenate([j * HW + (np.arange(HW) if PPI >= HW else rng.choice(HW, PPI, replace=False)) for j in range(B)]))
        zc = Z[b].reshape(-1, H[b])[sel]; w = mx.maximum(zc, 0)
        zn = PRE[b].reshape(-1, PRE[b].shape[-1])[sel] @ NULL[b][0] + NULL[b][1]; wn = mx.maximum(zn, 0)
        a = acc[b]; a["won"] += w.sum(0); a["on"] += (zc > 0).sum(0); a["nwon"] += wn.sum(0); a["non"] += (zn > 0).sum(0); a["rows"] += len(sel)
        for v, s in KEYS:
            sign = 1 if s == "pos" else -1
            par = mx.concatenate([((to_res(Z[p], p, b, v == "view", sign).reshape(-1, H[p])[sel] > 0) if s == "pos" else
                                   (to_res(Z[p], p, b, v == "view", sign).reshape(-1, H[p])[sel] <= TAU[p])).astype(mx.float32) for p in range(b)], 1)
            a[("n", v, s)] += wn.T @ par; a[("br", v, s)] += par.sum(0)
        mx.eval(a)
    if i % 500 == 0: print(i, round(time.time() - t0), "s", flush=True)

save = {}
for b in range(1, NB):
    a = acc[b]; save[f"{b}_nrate"] = np.array(a["non"]) / a["rows"]
    for v, s_ in KEYS:
        save[f"{b}_{v}_{s_}_nC"] = (np.array(a[("n", v, s_)]) / np.maximum(np.array(a["nwon"])[:, None], 1e-9)).astype(np.float16)
np.savez(os.path.join(HF, "imagenet", "m3_null.npz"), **save); print("null saved", round(time.time() - t0), "s")
