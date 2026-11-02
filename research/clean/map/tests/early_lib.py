"""Shared helpers for the early-band synthetic tests (b1 n351, b3 n623, b4 n499, b5 n135)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vision2"))
import numpy as np, mlx.core as mx
from PIL import Image, ImageDraw
from convnext import load_convnext, norm_img

_m = None


def model():
    global _m
    if _m is None: _m = load_convnext()
    return _m


def zmaps(b, n, imgs, bs=64):
    """fc1 pre-activation maps of neuron n in block b for uint8 images (N,224,224,3)."""
    m = model(); out = []
    for i in range(0, len(imgs), bs):
        x = m.stream(mx.array(norm_img(np.asarray(imgs[i:i + bs]))), upto=b)
        out.append(np.array(m.blocks[b].fc1(m.blocks[b].pre(x))[..., n]))
    return np.concatenate(out)


def stat(Z, margin=0.2):
    """max over the central region (drop a border to avoid padding artifacts)."""
    H = Z.shape[1]; k = int(H * margin)
    return Z[:, k:H - k, k:H - k].max((1, 2))


def line_img(angle, width=2, fg=0, bg=200, length=300, color=False):
    """One straight line through the centre. angle in degrees, 0 = horizontal, 90 = vertical."""
    im = Image.new("RGB", (224, 224), bg if color else (bg,) * 3); d = ImageDraw.Draw(im)
    a = np.deg2rad(angle); dx, dy = np.cos(a) * length / 2, -np.sin(a) * length / 2
    d.line([(112 - dx, 112 - dy), (112 + dx, 112 + dy)], fill=fg if color else (fg,) * 3, width=width)
    return np.array(im)


def grating(angle, period, duty=0.5, lo=0, hi=200, square=True):
    yy, xx = np.mgrid[0:224, 0:224].astype(np.float32) - 112
    a = np.deg2rad(angle)
    u = xx * np.sin(a) + yy * np.cos(a)        # stripes run along `angle` (0 = horizontal stripes, 90 = vertical)
    ph = (u / period) % 1.0
    v = (ph < duty).astype(np.float32) if square else 0.5 + 0.5 * np.cos(2 * np.pi * u / period)
    g = lo + (hi - lo) * v
    return np.repeat(g[..., None], 3, -1).clip(0, 255).astype(np.uint8)


def patch(rgb):
    return np.tile(np.array(rgb, np.uint8)[None, None], (224, 224, 1))


def multi(targets, imgs, bs=64):
    """targets: list of (block, neuron). Returns {(b,n): maps (N,H,W)} in one pass per batch."""
    m = model(); want = {}
    for b, n in targets: want.setdefault(b, []).append(n)
    out = {t: [] for t in targets}; top = max(want)
    for i in range(0, len(imgs), bs):
        x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(imgs[i:i + bs])))))
        for b in range(top + 1):
            if b > 0 and m.stage_of[b] != m.stage_of[b - 1]: x = m.down[m.stage_of[b] - 1](m.down_norm[m.stage_of[b] - 1](x))
            if b in want:
                z = np.array(m.blocks[b].fc1(m.blocks[b].pre(x))[..., want[b]])
                for j, n in enumerate(want[b]): out[(b, n)].append(z[..., j])
            x = m.blocks[b](x)
    return {t: np.concatenate(v) for t, v in out.items()}
