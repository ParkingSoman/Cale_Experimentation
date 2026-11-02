"""b12 n136: is the distinction the POLARITY of thin line-work? (dense dark arcs on light -> 74% of max; light
lines on dark -> strongly negative; the negative-pole crops are light straw, needles, grass, chain on darker ground.)
PREDICTION (before running): (1) the dense arc texture (N=256) inverted (light arcs on dark) gives mean z < -2.5.
(2) On the 24 most-negative natural images, inverting the image raises the image max by > +1.0 on median and the
mean z by > +1.5. (3) On the top-24 natural images, inverting lowers the mean z (median change < 0)."""
from lib4 import *
from PIL import Image, ImageDraw
b, n = 12, 136; MX = 2.83; rng = np.random.default_rng(0); X, y, ana = data(); A = acts()
def arcs(N=256):
    P = Image.new("RGB", (224, 224), (190, 190, 190)); D = ImageDraw.Draw(P)
    for _ in range(N):
        x, yy = rng.uniform(-10, 224, 2); s = rng.uniform(0, 360); D.arc([x, yy, x + 30, yy + 30], s, s + 200, fill=(40, 40, 40), width=4)
    return np.array(P)
I = np.stack([arcs() for _ in range(5)]); Z, Zi = zmap(b, n, I), zmap(b, n, 255 - I)
print(f"(1) dense arcs dark-on-light: max {Z.max((1,2)).mean():.2f} mean {Z.mean():.2f} | inverted: max {Zi.max((1,2)).mean():.2f} mean {Zi.mean():.2f}")
for lab, order in (("most-negative 24", np.argsort(A[f"min{b}"][:, n].astype(np.float32))[:24]), ("top 24", np.argsort(-A[f"max{b}"][:, n].astype(np.float32))[:24])):
    T = np.asarray(X[ana[order]]); z0, z1 = zmap(b, n, T), zmap(b, n, 255 - T)
    print(f"({'2' if 'neg' in lab else '3'}) {lab}: max {np.median(z0.max((1,2))):.2f} -> {np.median(z1.max((1,2))):.2f}; mean {np.median(z0.mean((1,2))):.2f} -> {np.median(z1.mean((1,2))):.2f}; min {np.median(z0.min((1,2))):.2f} -> {np.median(z1.min((1,2))):.2f}")
# random 300 images: correlation of per-image mean z between original and inverted
r = np.asarray(X[np.sort(rng.choice(ana, 300, replace=False))]); a0, a1 = zmap(b, n, r).mean((1, 2)), zmap(b, n, 255 - r).mean((1, 2))
print(f"random 300: corr(mean z, mean z inverted) = {np.corrcoef(a0, a1)[0,1]:.2f}; mean {a0.mean():.2f} -> {a1.mean():.2f}")
