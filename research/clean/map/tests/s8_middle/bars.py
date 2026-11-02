"""Bar sweep for b6 n1286 and candidates: bars of width w, length L, angle a (0 = vertical), on flat gray.
Value = max of the unit's map (pooled to 14x14) / unit's 99.9-pct image max (3000 set). Usage: python bars.py B,N ..."""
import sys, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", "..", "..", "vision2")); sys.path.insert(0, os.path.join(H, "..", "s6"))
import gallery as G
from judge_lib import maps
T = [tuple(map(int, a.split(","))) for a in sys.argv[1:]]; W = {}
for b, n in T: W.setdefault(b, []).append(n)
G._init(); S = G.scan(W); MX = {t: np.percentile(S[t][0], 99.9) for t in T}
def bar(w, L, ang, fg, bg):
    im = Image.new("RGB", (448, 448), bg); ImageDraw.Draw(im).rectangle([224 - w / 2, 224 - L / 2, 224 + w / 2, 224 + L / 2], fill=fg)
    return im.rotate(ang, resample=Image.BILINEAR, fillcolor=bg).crop((112, 112, 336, 336))
ST = {}
for w in (2, 4, 8, 16, 32):
    for L in (40, 100):
        ST[f"dark w{w} L{L} vert"] = bar(w, L, 0, (20, 20, 20), (170, 170, 170))
ST["dark w4 L100 horiz"] = bar(4, 100, 90, (20, 20, 20), (170, 170, 170)); ST["dark w4 L100 diag"] = bar(4, 100, 45, (20, 20, 20), (170, 170, 170))
ST["light w4 L100 vert"] = bar(4, 100, 0, (235, 235, 235), (60, 60, 60)); ST["red w4 L100 vert"] = bar(4, 100, 0, (210, 20, 20), (170, 170, 170))
ST["dark w4 L100 vert blur3"] = ST["dark w4 L100 vert"].filter(ImageFilter.GaussianBlur(3))
im = Image.new("RGB", (224, 224), (170, 170, 170)); ImageDraw.Draw(im).rectangle([112, 0, 224, 224], fill=(20, 20, 20)); ST["vertical step edge"] = im
im = Image.new("RGB", (224, 224), (170, 170, 170)); ImageDraw.Draw(im).rectangle([106, 106, 118, 118], fill=(20, 20, 20)); ST["dot 12"] = im
k = list(ST); Z = maps(np.stack([np.asarray(ST[x]) for x in k]), T)
def pool(z):
    f = z.shape[1] // 14; return z if f == 1 else z.reshape(len(z), 14, f, 14, f).max((2, 4))
print("stimulus".ljust(26) + "".join(f"b{b}n{n}".rjust(10) for b, n in T))
P = {t: pool(Z[t]).max((1, 2)) / MX[t] for t in T}
for i, x in enumerate(k): print(x.ljust(26) + "".join(f"{P[t][i]:10.2f}" for t in T))
