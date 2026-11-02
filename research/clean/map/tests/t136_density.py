"""b12 n136 negative pole: crowding. Same small dark arcs (r 15, width 4) on a light ground, N per image.
PREDICTION (before running): mean z falls steadily with N and is < -2 at N=256 (crowded marks = off pole), while
max z peaks at small N (isolated marks). Same for dark spots (leopard-like): mean < -2 at high density.
5 random images per N, averaged."""
from lib4 import *
from PIL import Image, ImageDraw
b, n = 12, 136; MX = 2.83; rng = np.random.default_rng(0)
def img(N, kind):
    P = Image.new("RGB", (224, 224), (190, 190, 190)); D = ImageDraw.Draw(P)
    for _ in range(N):
        x, y = rng.uniform(-10, 224, 2)
        if kind == "arc": s = rng.uniform(0, 360); D.arc([x, y, x + 30, y + 30], s, s + 200, fill=(40, 40, 40), width=4)
        else: r = rng.uniform(4, 9); D.ellipse([x, y, x + 2 * r, y + 1.5 * r], fill=(40, 40, 40))
    return np.array(P)
for kind in ("arc", "spot"):
    for N in (1, 4, 16, 64, 256, 1024):
        Z = zmap(b, n, np.stack([img(N, kind) for _ in range(5)]))
        print(f"{kind:4s} N={N:5d}  max {Z.max((1,2)).mean():5.2f} ({int(100*Z.max((1,2)).mean()/MX)}%)  mean {Z.mean():5.2f}")
