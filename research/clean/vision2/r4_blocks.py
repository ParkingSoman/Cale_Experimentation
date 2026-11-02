"""Block importance: each block's MLP branch replaced by its mean output (over calibration positions); sealed top-1.
Plus saliency check: per stream coordinate, mean |value| entering the head (after mean pool) vs median."""
import json
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
from acts import next_stream

m = load_convnext(); X, y = load_images(); ana, sea = split(y)
rank = np.zeros(len(y), int)
for c in range(1000):
    ix = np.where(y == c)[0]; rank[ix] = np.arange(len(ix))
cal = np.where(rank < 1)[0]; ev = np.where((rank >= 20) & (rank < 23))[0]          # 1,000 cal, 3,000 sealed
means = {}; pooled = []
for i in range(0, len(cal), 100):
    x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(X[cal[i:i + 100]])))))
    for b in range(18):
        if b > 0: x = next_stream(m, x, b - 1)
        blk = m.blocks[b]; br = blk.gamma * blk.fc2(nn.gelu(blk.fc1(blk.pre(x))))
        means.setdefault(b, []).append(np.array(br.mean((0, 1, 2)))); x = x + br
    pooled.append(np.array(x.mean((1, 2))))
P = np.concatenate(pooled); mag = np.abs(P).mean(0); top = np.argsort(-mag)[:8]
print("stream coords entering head, mean|x| top 8:", [(int(t), round(float(mag[t]), 2)) for t in top], "median %.3f" % np.median(mag))


class Mean(nn.Module):
    def __init__(self, v): super().__init__(); self.v = mx.array(v)
    def __call__(self, x): return x * 0 + self.v


def acc():
    return float(np.mean(np.concatenate([np.array(m(mx.array(norm_img(np.asarray(X[ev[i:i + 100]]))))).argmax(1) for i in range(0, len(ev), 100)]) == y[ev]))


out = {"intact": acc(), "saliency_top": [(int(t), float(mag[t])) for t in top], "median_mag": float(np.median(mag)), "blocks": {}}
print("intact %.4f" % out["intact"])
for b in range(18):
    blk = m.blocks[b]; orig = blk.__class__.__call__
    v = np.mean(means[b], 0)
    blk.__call__ = None
    saved = (blk.fc1, blk.fc2)
    class Patched(type(blk)): pass
    blk.__class__ = Patched; Patched.__call__ = lambda self, x, v=mx.array(v): x + v
    a = acc(); blk.__class__ = Patched.__mro__[1]
    out["blocks"][b] = a; print("block", b, "MLP -> mean: %.4f" % a, flush=True)
json.dump(out, open(os.path.join(RES, "vision-r4-blocks.json"), "w"), indent=1)
