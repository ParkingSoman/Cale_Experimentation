"""Per-image spatial max / min of fc1 pre-activation z for chosen blocks, analysis images (20 per class). Cached."""
import os, numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *

BLOCKS = (11, 12, 13, 14, 15, 16, 17)


def next_stream(m, x, b):
    """Stream after block b's output x -> stream entering block b+1 (downsample at a stage boundary)."""
    s, t = m.stage_of[b], m.stage_of[b + 1]
    return m.down[t - 1](m.down_norm[t - 1](x)) if t != s else x


def collect(m, X, blocks=BLOCKS, bs=100):
    mx_, mn_ = {b: [] for b in blocks}, {b: [] for b in blocks}
    for i in range(0, len(X), bs):
        x = m.stem_norm(m.stem(mx.array(norm_img(X[i:i + bs]))))
        for b in range(18):
            if b > 0: x = next_stream(m, x, b - 1)
            blk = m.blocks[b]
            if b in blocks:
                z = blk.fc1(blk.pre(x)); mx_[b].append(np.array(z.max((1, 2)), np.float16)); mn_[b].append(np.array(z.min((1, 2)), np.float16))
            x = blk(x)
    return {b: np.concatenate(mx_[b]) for b in blocks}, {b: np.concatenate(mn_[b]) for b in blocks}


def get_acts(m=None):
    f = os.path.join(HF, "imagenet", "acts_tiny_ana20k.npz")
    if not os.path.exists(f):
        X, y = load_images(); ana, _ = split(y); m = m or load_convnext()
        a, b = collect(m, X[ana])
        np.savez(f, **{f"max{k}": v for k, v in a.items()}, **{f"min{k}": v for k, v in b.items()})
    d = np.load(f)
    return {b: d[f"max{b}"].astype(np.float32) for b in BLOCKS}, {b: d[f"min{b}"].astype(np.float32) for b in BLOCKS}
