"""Shared helpers for the late-middle study (blocks 12-14): zmap on any images, big crop galleries from 20k images."""
import os, sys
_H4 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H4, "..", "..", "vision2"))
import numpy as np, mlx.core as mx
from PIL import Image, ImageDraw
from convnext import *

OUT = os.path.join(_H4, "out")
_m = None; _X = None; _y = None; _ana = None; _A = None


def M():
    global _m, _X, _y, _ana
    if _m is None:
        _m = load_convnext(); _X, _y = load_images(); _ana, _ = split(_y)
    return _m


def data():
    M(); return _X, _y, _ana


def acts():
    """per-image spatial max/min of z over the 20k analysis images (blocks 11-17)."""
    global _A
    if _A is None:
        d = np.load(os.path.join(HF, "imagenet", "acts_tiny_ana20k.npz")); _A = d
    return _A


def zmap(b, n, imgs, bs=64):
    """imgs uint8 (N,224,224,3) -> z maps (N,R,R) or (N,R,R,len(n)) if n is a list."""
    m = M(); out = []
    for i in range(0, len(imgs), bs):
        x = m.stream(mx.array(norm_img(np.asarray(imgs[i:i + bs]))), upto=b)
        out.append(np.array(m.blocks[b].fc1(m.blocks[b].pre(x))[..., n]))
    return np.concatenate(out)


def names():
    return [s.split(",")[0] for s in class_names()]


def overlay(im, z, alpha=110):
    s = max(abs(z).max(), 1e-6)
    hm = np.array(Image.fromarray(((np.clip(z / s, -1, 1) + 1) * 127.5).astype(np.uint8)).resize((im.shape[1], im.shape[0]), Image.BILINEAR)).astype(np.float32) / 255 * 2 - 1
    ov = im.astype(np.float32) * .6; ov[..., 0] += alpha * np.clip(hm, 0, 1); ov[..., 2] += alpha * np.clip(-hm, 0, 1)
    return np.clip(ov, 0, 255).astype(np.uint8)


def grid(tiles, labels, fname, cols=8, T=150, title=""):
    rows = (len(tiles) + cols - 1) // cols
    img = Image.new("RGB", (cols * (T + 4) + 4, rows * (T + 16) + 20), "white"); dr = ImageDraw.Draw(img); dr.text((4, 4), title, fill="black")
    for k, (t, l) in enumerate(zip(tiles, labels)):
        r, c = divmod(k, cols); img.paste(Image.fromarray(t).resize((T, T)), (4 + c * (T + 4), 20 + r * (T + 16)))
        dr.text((4 + c * (T + 4), 20 + r * (T + 16) + T + 1), l[:24], fill="black")
    fp = os.path.join(OUT, fname); img.save(fp); return fp


def crop_at(im, pos_rc, R, cs=136):
    r, c = pos_rc; s = 224 / R; cy, cx = (r + .5) * s, (c + .5) * s
    t = int(np.clip(cy - cs / 2, 0, 224 - cs)); l = int(np.clip(cx - cs / 2, 0, 224 - cs)); return im[t:t + cs, l:l + cs]
