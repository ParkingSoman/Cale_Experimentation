"""Crop viewer (essay microscope style): per neuron, the 8 strongest spots (crop centred there) and the 8 most
negative spots. gallery(rows, fname, title); rows = [(block, neuron, caption), ...]."""
import numpy as np, mlx.core as mx
from PIL import Image, ImageDraw
from convnext import *

CROP = {0: 40, 1: 72, 2: 136, 3: 224}           # px, roughly the receptive field per stage
_m = _X = _y = _idx = None


def _init(n_per_class=3):
    global _m, _X, _y, _idx
    if _m is None:
        _m = load_convnext(); _X, _y = load_images(); ana, _ = split(_y)
        _idx = np.concatenate([ana[_y[ana] == c][:n_per_class] for c in range(1000)])


def scan(want):
    """want: {block: [neurons]} -> {(b,n): (vals_max, pos_max, vals_min, pos_min)} per image."""
    _init(); out = {(b, n): [[], [], [], []] for b, ns in want.items() for n in ns}
    for i in range(0, len(_idx), 50):
        x = _m.stem_norm(_m.stem(mx.array(norm_img(np.asarray(_X[_idx[i:i + 50]])))))
        for b in range(max(want) + 1):
            if b > 0 and _m.stage_of[b] != _m.stage_of[b - 1]: x = _m.down[_m.stage_of[b] - 1](_m.down_norm[_m.stage_of[b] - 1](x))
            if b in want:
                z = np.array(_m.blocks[b].fc1(_m.blocks[b].pre(x))[..., want[b]]); B, Hh, W, k = z.shape; zf = z.reshape(B, Hh * W, k)
                for j, n in enumerate(want[b]):
                    o = out[(b, n)]; o[0].append(zf[:, :, j].max(1)); o[1].append(zf[:, :, j].argmax(1)); o[2].append(zf[:, :, j].min(1)); o[3].append(zf[:, :, j].argmin(1))
            x = _m.blocks[b](x)
    return {k: [np.concatenate(v) for v in vs] for k, vs in out.items()}


def crop(img_i, pos, b):
    st = _m.stage_of[b]; R = 56 // 2 ** st; r, c = divmod(int(pos), R); s = 224 / R; cs = CROP[st]
    cy, cx = (r + .5) * s, (c + .5) * s; t = int(np.clip(cy - cs / 2, 0, 224 - cs)); l = int(np.clip(cx - cs / 2, 0, 224 - cs))
    return Image.fromarray(np.asarray(_X[_idx[img_i]])[t:t + cs, l:l + cs]).resize((88, 88))


def gallery(rows, fname, title="", k=8, hide=False):
    _init(); want = {}
    for b, n, _ in rows:
        if n not in want.setdefault(b, []): want[b].append(n)
    S = scan(want); T = 88; W = T * (2 * k) + 60; Hh = (T + 26) * len(rows) + 40
    img = Image.new("RGB", (W, Hh), "white"); dr = ImageDraw.Draw(img); dr.text((8, 6), title, fill="black"); yy = 34
    names = [n.split(",")[0] for n in class_names()]
    for b, n, cap in rows:
        vm, pm, vn, pn = S[(b, n)]
        dr.text((8, yy - 12), (cap if hide else f"block {b} n{n}  {cap}   [+ strongest spots | - most negative spots]"), fill="black")
        for j, i in enumerate(np.argsort(-vm)[:k]):
            img.paste(crop(i, pm[i], b), (8 + j * T, yy)); dr.text((8 + j * T, yy + T), names[_y[_idx[i]]][:13], fill="darkgreen")
        for j, i in enumerate(np.argsort(vn)[:k]):
            img.paste(crop(i, pn[i], b), (40 + (k + j) * T, yy)); dr.text((40 + (k + j) * T, yy + T), names[_y[_idx[i]]][:13], fill="darkred")
        yy += T + 26
    fp = os.path.join(HERE, "figs", fname); img.save(fp); return fp
