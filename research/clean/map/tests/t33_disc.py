"""b14 n33 size + colour test. Galleries: small warm (orange/red/yellow) spots such as beaks, tongues, combs, held
food; the off pole has LARGE warm surfaces (bread, squash, pie). Hue rotation halves it (colour is ~half the job).
PREDICTION (before running): an orange disc pasted at the centre of 20 ordinary photos (low response) drives the
unit to >50% of max (>2.8, median) at diameter 24-48 px; at 160 px it is < half of the best size; a blue disc of the
best size stays <25% of max; a gray disc <25%."""
from lib4 import *
from PIL import Image, ImageDraw
b, n = 14, 33; MX = 5.64; X, y, ana = data(); A = acts(); rng = np.random.default_rng(0)
v = A[f"max{b}"][:, n].astype(np.float32); pool = np.where(v < np.percentile(v, 40))[0]; bgi = rng.choice(pool, 20, replace=False)
BG = np.asarray(X[ana[np.sort(bgi)]])
print(f"backgrounds alone: median max {np.median(zmap(b, n, BG).max((1,2))):.2f}")
for col_name, col in (("orange", (235, 120, 30)), ("red", (210, 30, 30)), ("yellow", (240, 210, 40)), ("brown", (130, 80, 40)), ("blue", (40, 90, 220)), ("green", (50, 170, 60)), ("gray", (128, 128, 128))):
    res = []
    for D in (8, 16, 24, 32, 48, 64, 96, 160):
        I = []
        for im in BG:
            P = Image.fromarray(im.copy()); dr = ImageDraw.Draw(P); dr.ellipse([112 - D / 2, 112 - D / 2, 112 + D / 2, 112 + D / 2], fill=col); I.append(np.array(P))
        Z = zmap(b, n, np.stack(I)); res.append(f"{D}:{int(100*np.median(Z[:, 5:9, 5:9].max((1,2)))/MX)}")
    print(f"{col_name:6s} disc diameter:%max (centre 4x4 max, median of 20) ", " ".join(res))
