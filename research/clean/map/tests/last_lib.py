"""Shared helpers for the last-stage study (blocks 15-17: 15 n16, 16 n384, 16 n26, 16 n2062, 17 n2283)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vision2"))
import numpy as np, mlx.core as mx
from PIL import Image, ImageDraw
from convnext import load_convnext, norm_img, load_images, split, class_names, HF

_m = None; _X = _y = _ana = None; _names = None
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs"); os.makedirs(OUT, exist_ok=True)


def model():
    global _m
    if _m is None: _m = load_convnext()
    return _m


def data():
    global _X, _y, _ana
    if _X is None: _X, _y = load_images(); _ana, _ = split(_y)
    return _X, _y, _ana


def names():
    global _names
    if _names is None: _names = [s.split(",")[0] for s in class_names()]
    return _names


def zmaps(b, n, imgs, bs=50):
    """fc1 pre-activation maps (N,7,7) of neuron n (int or list) in block b for uint8 images (N,224,224,3)."""
    m = model(); out = []
    for i in range(0, len(imgs), bs):
        x = m.stream(mx.array(norm_img(np.asarray(imgs[i:i + bs]))), upto=b)
        out.append(np.array(m.blocks[b].fc1(m.blocks[b].pre(x))[..., n]))
    return np.concatenate(out)


def cache():
    """per-image spatial max/min over the 20k analysis images (20 per class), blocks 11-17."""
    d = np.load(os.path.join(HF, "imagenet", "acts_tiny_ana20k.npz"), mmap_mode="r"); return d


def amax(b, n):
    return cache()[f"max{b}"][:, n].astype(np.float32)


def amin(b, n):
    return cache()[f"min{b}"][:, n].astype(np.float32)


def labels_ana():
    X, y, ana = data(); return y[ana]


def class_mean(v):
    ya = labels_ana(); return np.bincount(ya, weights=v, minlength=1000) / np.bincount(ya, minlength=1000)


def top_classes(v, k=15, neg=False):
    c = class_mean(v); o = np.argsort(c if neg else -c)[:k]; nm = names()
    return [(nm[i], round(float(c[i]), 2)) for i in o]


def heat_sheet(b, n, idx, fname, title=""):
    """big heat maps (red +, blue -) on chosen dataset images (global indices)."""
    X, y, _ = data(); imgs = np.stack([np.asarray(X[i]) for i in idx]); Z = zmaps(b, n, imgs); s = max(np.abs(Z).max(), 1e-6)
    W = 230 * min(len(idx), 6); H = 240 * ((len(idx) + 5) // 6) + 20; sheet = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(sheet)
    d.text((4, 2), title, fill="black")
    for j, i in enumerate(idx):
        im = imgs[j].astype(np.float32); z = Z[j]
        hm = np.array(Image.fromarray(((np.clip(z / s, -1, 1) + 1) * 127.5).astype(np.uint8)).resize((224, 224), Image.BILINEAR)).astype(np.float32) / 255 * 2 - 1
        ov = im * .6; ov[..., 0] += 100 * np.clip(hm, 0, 1); ov[..., 2] += 100 * np.clip(-hm, 0, 1)
        x0, y0 = 230 * (j % 6), 20 + 240 * (j // 6)
        sheet.paste(Image.fromarray(np.clip(ov, 0, 255).astype(np.uint8)), (x0, y0))
        d.text((x0 + 2, y0 + 224), f"{names()[y[i]][:18]} max {z.max():.1f} min {z.min():.1f}", fill="black")
    fp = os.path.join(OUT, fname); sheet.save(fp); return fp
