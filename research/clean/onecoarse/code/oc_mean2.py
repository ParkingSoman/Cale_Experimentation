"""Intersection images, sharper (README rule 7). For each member and pole: the 60 strongest cells over the 19,000
held + test photos (positive: max of value - offset; negative: input-only min), windows 4 cells wide aligned on
the cell. The windows are split into 3 sub-patterns (k-means on their small edge maps, a branch check for recursive
generalization), and each group's colour mean and edge mean is shown with its size.
-> results/means2_<member>.png ; usage: python3 oc_mean2.py [members]"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "..")
D = np.load(os.path.join(HERE, "..", "results", "oc_split.npz"))
IXT = np.load(os.path.join(CLEAN, "curves", "cache", "r18", "ix_train.npy"))
X = np.load(os.path.join(CLEAN, "hf", "imagenet", "val_224.npy"), mmap_mode="r")
S, N, K = 150, 60, 3


def win(r, cell, grid):
    c = 224 // grid; w = 4 * c
    img = np.pad(np.asarray(X[int(IXT[D["ix"][r]])], np.float32), ((w, w), (w, w), (0, 0)), mode="reflect")
    rr, kk = divmod(int(cell), grid)
    cy, cx = int((rr + .5) * c) + w, int((kk + .5) * c) + w
    return img[cy - w // 2:cy + w // 2, cx - w // 2:cx + w // 2]


def edges(a):
    gy, gx = np.gradient(a.mean(2)); return np.hypot(gx, gy)


def small(e, n=16):
    h = e.shape[0] // n
    return e[:h * n, :h * n].reshape(n, h, n, h).mean((1, 3)).ravel()


def kmeans(F, k, seed=0):
    rng = np.random.default_rng(seed)
    F = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-6)
    C = F[rng.choice(len(F), k, replace=False)]
    for _ in range(50):
        lab = ((F[:, None] - C[None]) ** 2).sum(2).argmin(1)
        C = np.stack([F[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    return lab


def tile(a, grey=False):
    if grey:
        a = np.repeat(np.clip(a / (np.quantile(a, .99) + 1e-6) * 255, 0, 255)[..., None], 3, 2)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((S, S))


def member(n):
    grid = 14 if n == "p" else 7
    rows = []
    for side, v, am in (("positive", D[f"{n}_max"], D[f"{n}_amax"]), ("negative", -D[f"{n}_min"], D[f"{n}_amin"])):
        top = np.argsort(-v)[:N]
        W = np.stack([win(r, am[r], grid) for r in top]); E = np.stack([edges(w) for w in W])
        lab = kmeans(np.stack([small(e) for e in E]), K)
        groups = [("all", np.ones(N, bool))] + [(f"group {j + 1}", lab == j) for j in np.argsort(-np.bincount(lab, minlength=K))]
        rows.append((side, [(g, int(m.sum()), W[m].mean(0), E[m].mean(0)) for g, m in groups if m.any()]))
    img = Image.new("RGB", (8 * S + 10, 2 * (S + 20) + 10), "white")
    dr = ImageDraw.Draw(img)
    for i, (side, groups) in enumerate(rows):
        y = i * (S + 20) + 4
        for j, (g, cnt, cm, em) in enumerate(groups):
            x = j * 2 * S
            dr.text((x + 2, y), f"{n} {side} {g} (n={cnt})", fill="black")
            img.paste(tile(cm), (x, y + 14)); img.paste(tile(em, grey=True), (x + S, y + 14))
    p = os.path.join(HERE, "..", "results", f"means2_{n}.png")
    img.save(p)
    return p


if __name__ == "__main__":
    for n in (sys.argv[1:] or ["p", "k275", "k297", "k97", "k305", "k163", "k23", "c101"]):
        print(member(n))
