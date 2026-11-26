"""Word-free 'intersection' images (essay footnote 4, a cheap version): for each member, the pixel mean of its 30
positive band windows and its 30 negative band windows, aligned on the cell (window = 4 cells wide, centred), plus
the grey-level mean edge map (|gradient|) of the same windows, which shows shared SHAPE even when colours differ.
-> results/means.png (rows: members; columns: pos colour, pos edges, neg colour, neg edges)"""
import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "..")
D = np.load(os.path.join(HERE, "..", "results", "oc_split.npz"))
IXT = np.load(os.path.join(CLEAN, "curves", "cache", "r18", "ix_train.npy"))
X = np.load(os.path.join(CLEAN, "hf", "imagenet", "val_224.npy"), mmap_mode="r")
S = 160
NAMES = {"p": "parent s3:69", "k275": "n275", "k297": "n297", "k97": "n97", "k305": "n305", "k163": "n163", "k23": "n23", "c101": "coarse n101"}


def win(r, cell, grid):
    c = 224 // grid; w = 4 * c
    img = np.pad(np.asarray(X[int(IXT[D["ix"][r]])], np.float32), ((w, w), (w, w), (0, 0)), mode="reflect")
    rr, kk = divmod(int(cell), grid)
    cy, cx = int((rr + .5) * c) + w, int((kk + .5) * c) + w
    return img[cy - w // 2:cy + w // 2, cx - w // 2:cx + w // 2]


def edges(a):
    g = a.mean(2); gy, gx = np.gradient(g); return np.hypot(gx, gy)


def to_img(a, grey=False):
    if grey:
        a = a / (np.quantile(a, .99) + 1e-6) * 255
        a = np.repeat(np.clip(a, 0, 255)[..., None], 3, 2)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((S, S))


rows = list(NAMES)
img = Image.new("RGB", (4 * S + 120, len(rows) * (S + 6) + 20), "white")
dr = ImageDraw.Draw(img)
for k, t in enumerate(("positive mean", "positive edges", "negative mean", "negative edges")):
    dr.text((120 + k * S + 4, 4), t, fill="black")
for i, n in enumerate(rows):
    grid = 14 if n == "p" else 7
    y = 20 + i * (S + 6)
    dr.text((4, y + S // 2), NAMES[n], fill="black")
    for j, (side, am) in enumerate((("bpos", D[f"{n}_amax"]), ("bneg", D[f"{n}_amin"]))):
        W = np.stack([win(r, am[r], grid) for r in D[f"{n}_{side}"]])
        img.paste(to_img(W.mean(0)), (120 + 2 * j * S, y))
        img.paste(to_img(np.stack([edges(w) for w in W]).mean(0), grey=True), (120 + (2 * j + 1) * S, y))
p = os.path.join(HERE, "..", "results", "means.png")
img.save(p)
print(p)
