"""Pixel measures of the verb readings (Mac, numpy). For each member's 30 positive and 30 negative band cells
(oc_split.npz), and a random cell of the same photo, on the grey 224 px photo, in a window twice the cell
(stage 4: cell 32 px, window 64 px; parent: cell 16 px, window 32 px), centred on the cell:
  radial    mean |cos(gradient, direction from the centre)|, weighted by gradient magnitude (low = lines radiating)
  isolation |mean(cell) - mean(ring)| / std(window); ringtex = ring texture / cell texture (mean |Laplacian|)
  repeat    highest normalised autocorrelation of the window at shifts of 4 px or more
  crisp     mean |Laplacian| / mean |gradient|
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "..")
D = np.load(os.path.join(HERE, "..", "results", "oc_split.npz"))
IXT = np.load(os.path.join(CLEAN, "curves", "cache", "r18", "ix_train.npy"))
X = np.load(os.path.join(CLEAN, "hf", "imagenet", "val_224.npy"), mmap_mode="r")


def window(g, cell, grid):
    c = 224 // grid; w = 2 * c
    r, k = divmod(int(cell), grid)
    cy, cx = int((r + .5) * c), int((k + .5) * c)
    y0, x0 = min(max(cy - w // 2, 0), 224 - w), min(max(cx - w // 2, 0), 224 - w)
    return g[y0:y0 + w, x0:x0 + w], (cy - y0, cx - x0), c


def measures(g, cell, grid):
    W, (cy, cx), c = window(g, cell, grid)
    gy, gx = np.gradient(W); mag = np.hypot(gx, gy) + 1e-6
    yy, xx = np.mgrid[:W.shape[0], :W.shape[1]]; ry, rx = yy - cy, xx - cx; rn = np.hypot(rx, ry) + 1e-6
    radial = float((np.abs(gx * rx + gy * ry) / (mag * rn) * mag).sum() / mag.sum())
    inner = (np.abs(ry) < c / 2) & (np.abs(rx) < c / 2)
    lap = np.abs(4 * W[1:-1, 1:-1] - W[:-2, 1:-1] - W[2:, 1:-1] - W[1:-1, :-2] - W[1:-1, 2:])
    inn = inner[1:-1, 1:-1]
    iso = float(abs(W[inner].mean() - W[~inner].mean()) / (W.std() + 1e-6))
    ringtex = float(lap[~inn].mean() / (lap[inn].mean() + 1e-6))
    Z = W - W.mean(); F = np.fft.fft2(Z); ac = np.real(np.fft.ifft2(F * np.conj(F))); ac = np.fft.fftshift(ac) / (ac.max() + 1e-6)
    h = ac.shape[0] // 2; dy, dx = np.mgrid[:ac.shape[0], :ac.shape[1]]
    repeat = float(ac[np.hypot(dy - h, dx - h) >= 4].max())
    crisp = float(lap.mean() / (mag.mean() + 1e-6))
    return np.array([radial, iso, ringtex, repeat, crisp])


rng = np.random.default_rng(0)
names = ["radial", "isolation", "ringtex", "repeat", "crisp"]
print(f"{'member':7s} {'side':4s} " + " ".join(f"{n:>17s}" for n in names) + "   (value / random)")
for n in ["p", "k275", "k297", "k97", "k305", "k163", "k23", "c101"]:
    grid = 14 if n == "p" else 7
    for side, rows, am in (("pos", D[f"{n}_bpos"], D[f"{n}_amax"]), ("neg", D[f"{n}_bneg"], D[f"{n}_amin"])):
        v, vr = [], []
        for r in rows:
            g = np.asarray(X[int(IXT[D["ix"][r]])], np.float32).mean(2)
            v.append(measures(g, am[r], grid)); vr.append(measures(g, rng.integers(grid * grid), grid))
        v, vr = np.mean(v, 0), np.mean(vr, 0)
        print(f"{n:7s} {side:4s} " + " ".join(f"{a:8.3f} /{b:7.3f}" for a, b in zip(v, vr)))
