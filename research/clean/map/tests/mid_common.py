"""Shared helpers for the middle-band study (blocks 7, 8, 10, 11). Pre-activation (fc1 output) maps, 14x14."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vision2"))
import numpy as np, mlx.core as mx
from PIL import Image, ImageDraw
from convnext import *
import gallery as G

NEUR = [(7, 604), (8, 4), (10, 1316), (11, 1031)]
OUTD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUTD, exist_ok=True)


def zmaps(imgs, targets=NEUR, bs=50):
    """imgs uint8 (N,224,224,3) -> {(b,n): (N,14,14)} pre-activation maps."""
    G._init(); m = G._m; out = {t: [] for t in targets}; want = {}
    for b, n in targets: want.setdefault(b, []).append(n)
    for i in range(0, len(imgs), bs):
        x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(imgs[i:i + bs])))))
        for b in range(max(want) + 1):
            if b > 0 and m.stage_of[b] != m.stage_of[b - 1]: x = m.down[m.stage_of[b] - 1](m.down_norm[m.stage_of[b] - 1](x))
            if b in want:
                z = np.array(m.blocks[b].fc1(m.blocks[b].pre(x))[..., want[b]])
                for j, n in enumerate(want[b]): out[(b, n)].append(z[..., j])
            x = m.blocks[b](x)
    return {t: np.concatenate(v) for t, v in out.items()}


def heat(im, z, size=160):
    im = np.asarray(im).astype(np.float32); s = max(abs(z).max(), 1e-6)
    hm = np.array(Image.fromarray(((np.clip(z / s, -1, 1) + 1) * 127.5).astype(np.uint8)).resize((224, 224), Image.BILINEAR)).astype(np.float32) / 255 * 2 - 1
    ov = im * .55; ov[..., 0] += 115 * np.clip(hm, 0, 1); ov[..., 2] += 115 * np.clip(-hm, 0, 1)
    return Image.fromarray(np.clip(ov, 0, 255).astype(np.uint8)).resize((size, size))


def sheet(cells, fname, ncol=8, size=160, title=""):
    """cells: list of (PIL image, caption)."""
    nr = (len(cells) + ncol - 1) // ncol; W = ncol * (size + 4) + 4; H = 20 + nr * (size + 16)
    img = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(img); dr.text((4, 4), title, fill="black")
    for k, (p, cap) in enumerate(cells):
        r, c = divmod(k, ncol); x, y = 4 + c * (size + 4), 20 + r * (size + 16)
        img.paste(p.resize((size, size)), (x, y)); dr.text((x, y + size + 1), cap[:28], fill="black")
    fp = os.path.join(OUTD, fname); img.save(fp); return fp
