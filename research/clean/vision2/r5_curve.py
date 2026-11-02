"""Cut curves per block (cut_ls) vs whole-branch mean replacement. Stochastic-depth check, VISION-LOG R4."""
import json, sys
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
from acts import next_stream

m = load_convnext(); X, y = load_images()
rank = np.zeros(len(y), int)
for c in range(1000):
    ix = np.where(y == c)[0]; rank[ix] = np.arange(len(ix))
cal = np.where(rank < 1)[0]; ev = np.where((rank >= 20) & (rank < 23))[0]
rng = np.random.default_rng(0)


def cal_rows(b, bs=100):
    xs = []
    for i in range(0, len(cal), bs):
        x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(X[cal[i:i + bs]])))))
        for j in range(b):
            x = m.blocks[j](x); x = next_stream(m, x, j)
        u = np.array(m.blocks[b].pre(x)); n, h, w, c = u.shape; u = u.reshape(n, h * w, c)
        pick = rng.integers(0, h * w, (n, 49)); xs.append(np.take_along_axis(u, pick[:, :, None], 1).reshape(-1, c))
    return np.concatenate(xs).astype(np.float64)


def acc():
    return float(np.mean(np.concatenate([np.array(m(mx.array(norm_img(np.asarray(X[ev[i:i + 100]]))))).argmax(1) for i in range(0, len(ev), 100)]) == y[ev]))


def ls(F, T, lam=1e-4):
    A = np.c_[F, np.ones(len(F))]; G = A.T @ A; G += np.eye(len(G)) * lam * np.trace(G) / len(G)
    S = np.linalg.solve(G, A.T @ T); return S[:-1].T, S[-1]


from scipy.special import erf
gelu = lambda z: 0.5 * z * (1 + erf(z / np.sqrt(2)))
out = {"intact": acc()}; print("intact %.4f" % out["intact"], flush=True)
for b in map(int, sys.argv[1].split(",")):
    blk = m.blocks[b]; U = cal_rows(b)
    W1, b1 = np.array(blk.fc1.weight, np.float64), np.array(blk.fc1.bias, np.float64)
    g = np.array(blk.gamma, np.float64); D = np.array(blk.fc2.weight, np.float64) * g[:, None]; cb = g * np.array(blk.fc2.bias, np.float64)
    H = gelu(U @ W1.T + b1); Y = H @ D.T + cb; n = W1.shape[0]
    order = np.argsort(-(np.linalg.norm(H, axis=0) * np.linalg.norm(D, axis=0)))
    f1, f2 = blk.fc1, blk.fc2; row = {}
    for frac in (.75, .5, .25, .125, .03, 0.0):
        K = max(1, int(round(frac * n))) if frac > 0 else 0
        if K:
            keep = np.sort(order[:K]); Dk, bk = ls(H[:, keep], Y)
            blk.fc1 = nn.Linear(W1.shape[1], K); blk.fc1.weight = mx.array(W1[keep].astype(np.float32)); blk.fc1.bias = mx.array(b1[keep].astype(np.float32))
        else:   # whole branch MLP -> its mean (fc1 kept 1 dead unit)
            Dk, bk = np.zeros((D.shape[0], 1)), Y.mean(0)
            blk.fc1 = nn.Linear(W1.shape[1], 1); blk.fc1.weight = mx.zeros((1, W1.shape[1])); blk.fc1.bias = mx.full((1,), -10.0)
        blk.fc2 = nn.Linear(Dk.shape[1], D.shape[0]); blk.fc2.weight = mx.array((Dk / g[:, None]).astype(np.float32)); blk.fc2.bias = mx.array((bk / g).astype(np.float32))
        row[frac] = acc(); blk.fc1, blk.fc2 = f1, f2
    out[b] = row; print("block", b, " ".join("%g:%.4f" % (k, v) for k, v in row.items()), flush=True)
json.dump(out, open(os.path.join(RES, "vision-r5-curves.json"), "w"), indent=1)
