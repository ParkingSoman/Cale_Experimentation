"""Vision lab analysis helpers (rules: research/clean/LINEAGE-LOG.md Round 3)."""
import json, os
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from net import *

F2C = None


def f2c(d):
    m = np.zeros(100, int)
    for f, c in zip(d["ytr"], d["ctr"]): m[f] = c
    return m


def stream_inputs(model, x, b, bs=500):
    """fc1 input u = ln(dw(stream)) at block b, shape (N, 8, 8, C)."""
    out = []
    for i in range(0, len(x), bs):
        h = model.ln0(model.stem(mx.array(x[i:i + bs])))
        for j in range(b):
            h = model.blocks[j](h)
        blk = model.blocks[b]; out.append(np.array(blk.ln(blk.dw(h))))
    return np.concatenate(out)


def gelu_np(z):
    from scipy.special import erf
    return 0.5 * z * (1 + erf(z / np.sqrt(2)))


def mlp_np(blk):
    return (np.array(blk.fc1.weight, np.float64), np.array(blk.fc1.bias, np.float64),
            np.array(blk.fc2.weight, np.float64), np.array(blk.fc2.bias, np.float64))


def neuron_max(model, x, b, pool="max"):
    U = stream_inputs(model, x, b).astype(np.float64)
    W1, b1, _, _ = mlp_np(model.blocks[b])
    H = gelu_np(U @ W1.T + b1)
    H = H.reshape(len(x), -1, H.shape[-1])
    return H.max(1) if pool == "max" else H.mean(1)


def ls_out(Hm, Y, ridge=1e-4):
    A = np.c_[Hm, np.ones(len(Hm))]; G = A.T @ A; G += np.eye(len(G)) * ridge * np.trace(G) / len(G)
    S = np.linalg.solve(G, A.T @ Y); return S[:-1].T, S[-1]


def set_mlp(blk, W1, b1, W2, b2):
    C, K = W1.shape[1], W1.shape[0]
    blk.fc1 = nn.Linear(C, K); blk.fc2 = nn.Linear(K, C)
    blk.fc1.weight, blk.fc1.bias = mx.array(W1.astype(np.float32)), mx.array(b1.astype(np.float32))
    blk.fc2.weight, blk.fc2.bias = mx.array(W2.astype(np.float32)), mx.array(b2.astype(np.float32))


def calib(d, n=5000):
    return d["xtr"][np.sort(np.random.default_rng(777).permutation(50000)[:n])]


def acc(model, d, idx):
    a = accuracy(logits(model, d["xte"][idx]), d["yte"][idx], d["cte"][idx], f2c(d))
    return a


def boot_diff(a, b, n=2000, seed=0):
    d = a.astype(float) - b.astype(float); r = np.random.default_rng(seed)
    m = np.array([d[r.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return [float(d.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def save(name, obj):
    p = os.path.join(RES, f"vision-{name}.json"); json.dump(obj, open(p, "w"), indent=1); return p
