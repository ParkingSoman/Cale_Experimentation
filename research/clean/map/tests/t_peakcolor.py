"""Descriptive: colour of the 32x32 pixels at the peak cell, top-200 images vs 200 random images' peak cells.
Reports hue histogram (12 bins of 30 deg, saturated pixels only S>0.25), median saturation, value."""
from lib4 import *
from PIL import Image
X, y, ana = data(); A = acts(); rng = np.random.default_rng(0)
def stats(T, Z):
    H = []; Sx = []; Vx = []
    for t, z in zip(T, Z):
        r, c = np.unravel_index(np.argmax(z), z.shape); cy, cx = int((r + .5) * 16), int((c + .5) * 16)
        p = t[max(cy - 16, 0):cy + 16, max(cx - 16, 0):cx + 16]; h = np.array(Image.fromarray(p).convert("HSV")).reshape(-1, 3) / 255.
        H.append(h[h[:, 1] > .25, 0]); Sx.append(np.median(h[:, 1])); Vx.append(np.median(h[:, 2]))
    H = np.concatenate(H); hist = np.histogram(H * 360, bins=12, range=(0, 360))[0]
    return (hist / max(hist.sum(), 1) * 100).astype(int), np.median(Sx), np.median(Vx), len(H)
for b, n in ((14, 33), (14, 831), (12, 136)):
    v = A[f"max{b}"][:, n].astype(np.float32); top = np.argsort(-v)[:200]; rnd = rng.choice(len(v), 200, replace=False)
    for lab, idx in (("top200", top), ("random", rnd)):
        T = np.asarray(X[ana[np.sort(idx)]]); h, s, vv, k = stats(T, zmap(b, n, T))
        print(f"b{b} n{n} {lab:7s} hue% [0,30,..330]: {list(h)}  medS {s:.2f} medV {vv:.2f}")
