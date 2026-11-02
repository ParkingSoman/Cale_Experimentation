"""Cross-model crop sheets. family_sheet(p, c, small_name, s_neuron, fname): rows for the tiny parent, tiny child and
the small-model neuron, plus the small neuron's response (as % of its max) on the child's top spots and on the
parent-only spots (parent high, child off)."""
import numpy as np, mlx.core as mx, mlx.nn as nn
from PIL import Image, ImageDraw
from convnext import *
CFG = {"tiny": ((3, 3, 9, 3), (96, 192, 384, 768)), "atto": ((2, 2, 6, 2), (40, 80, 160, 320)), "nano": ((2, 2, 8, 2), (80, 160, 320, 640))}
CROP = {0: 40, 1: 72, 2: 136, 3: 224}; T = 88
_M = {}; _X = _y = _idx = None


def _init():
    global _X, _y, _idx
    if _X is None:
        _X, _y = load_images(); ana, _ = split(_y); _idx = np.concatenate([ana[_y[ana] == c][:3] for c in range(1000)])


def model(name):
    if name not in _M: _M[name] = load_convnext(name, *CFG[name])
    return _M[name]


def maps(name, want):
    """want: {block: [neurons]} -> {(b, n): hidden maps (N, R*R)} on the 3,000 gallery images."""
    _init(); m = model(name); out = {(b, n): [] for b, ns in want.items() for n in ns}; ST = m.stage_of
    for i in range(0, len(_idx), 50):
        x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(_X[_idx[i:i + 50]])))))
        for b in range(max(want) + 1):
            if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
            if b in want:
                h = np.array(nn.gelu(m.blocks[b].fc1(m.blocks[b].pre(x)))[..., want[b]]); B, R, C, k = h.shape
                for j, n in enumerate(want[b]): out[(b, n)].append(h[..., j].reshape(B, R * C))
            x = m.blocks[b](x)
    return {k: np.concatenate(v) for k, v in out.items()}


def crop(i, p, st):
    R = 56 // 2 ** st; r, c = divmod(int(p), R); s = 224 / R; cs = CROP[st]; cy, cx = (r + .5) * s, (c + .5) * s
    t = int(np.clip(cy - cs / 2, 0, 224 - cs)); l = int(np.clip(cx - cs / 2, 0, 224 - cs))
    return Image.fromarray(np.asarray(_X[_idx[i]])[t:t + cs, l:l + cs]).resize((T, T))


def spots(score, k, per_img=True):
    """top-k (image, position) by score (N, P); one spot per image."""
    best = score.argmax(1); v = score[np.arange(len(score)), best]; ims = np.argsort(-v)[:k]; return [(i, best[i]) for i in ims]


def family_sheet(p, c, small, s, fname, title="", k=8):
    _init(); names = [n.split(",")[0] for n in class_names()]
    want = {}
    for b, n in (p, c): want.setdefault(b, []).append(n)
    Ht = maps("tiny", want); Hs = maps(small, {s[0]: [s[1]]})[tuple(s)]
    hp, hc = Ht[tuple(p)], Ht[tuple(c)]; st = model("tiny").stage_of[c[0]]
    nz = lambda a: a / max(a.max(), 1e-6)
    rows = [(f"tiny parent block {p[0]} n{p[1]}: strongest spots", spots(hp, k), hp),
            (f"tiny child block {c[0]} n{c[1]}: strongest spots", spots(hc, k), hc),
            (f"{small} block {s[0]} n{s[1]}: strongest spots", spots(Hs, k), Hs),
            (f"{small} n{s[1]} on the CHILD's strongest spots (value = % of its max)", spots(hc, k), Hs),
            (f"{small} n{s[1]} on PARENT-ONLY spots (parent high, child off)", spots(nz(hp) - 3 * nz(np.maximum(hc, 0)), k), Hs),
            (f"tiny child n{c[1]} on the same parent-only spots", spots(nz(hp) - 3 * nz(np.maximum(hc, 0)), k), hc)]
    W = T * k + 16; Hh = (T + 30) * len(rows) + 36; img = Image.new("RGB", (W, Hh), "white"); dr = ImageDraw.Draw(img); dr.text((8, 6), title, fill="black"); yy = 34
    for cap, sp, val in rows:
        dr.text((8, yy - 12), cap, fill="black"); mx_ = max(val.max(), 1e-6)
        for j, (i, q) in enumerate(sp):
            img.paste(crop(i, q, st), (8 + j * T, yy)); dr.text((8 + j * T, yy + T), f"{names[_y[_idx[i]]][:9]} {100 * val[i, q] / mx_:.0f}%", fill="darkgreen")
        yy += T + 30
    fp = os.path.join(HERE, "figs", fname); img.save(fp); return fp
