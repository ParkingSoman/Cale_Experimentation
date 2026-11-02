"""b12 n136. Galleries: + = one clean thin stroke (cable, pole, wire, slat) on a smooth ground; - = dense busy organic
texture (straw, fur, scales). Fingerprint: blur3 0.43, zoomin 0.43, zoomout 0.50 (scale-tuned), invert 0.77.
PREDICTION (before running): a single straight line on plain gray reaches >50% of max (>1.4) at some width between
2 and 8 px, both polarities, at 0/45/90 deg; widths of 1 px and >=24 px stay below 25%. The SAME line drawn 150 times
at random as clutter goes negative (< -2) at the centre."""
from lib4 import *
from PIL import Image, ImageDraw
import math
b, n = 12, 136; MX = 2.83; rng = np.random.default_rng(0)
def line(w, ang, fg, bg, L=150):
    P = Image.new("RGB", (224, 224), bg); D = ImageDraw.Draw(P); a = math.radians(ang); dx, dy = L / 2 * math.cos(a), L / 2 * math.sin(a)
    D.line([(112 - dx, 112 - dy), (112 + dx, 112 + dy)], fill=fg, width=w); return np.array(P)
def clutter(w, fg, bg, k=150, L=40):
    P = Image.new("RGB", (224, 224), bg); D = ImageDraw.Draw(P)
    for _ in range(k):
        x, y = rng.uniform(0, 224, 2); a = rng.uniform(0, math.pi); D.line([(x, y), (x + L * math.cos(a), y + L * math.sin(a))], fill=fg, width=w)
    return np.array(P)
W = [1, 2, 3, 4, 6, 8, 12, 16, 24, 40]
for pol, fg, bg in (("dark on light", (40, 40, 40), (190, 190, 190)), ("light on dark", (220, 220, 220), (70, 70, 70))):
    for ang in (0, 45, 90):
        Z = zmap(b, n, np.stack([line(w, ang, fg, bg) for w in W]))
        print(f"{pol:14s} {ang:2d}deg  width->%max:", " ".join(f"{w}:{int(100*z.max()/MX)}" for w, z in zip(W, Z)))
    C = zmap(b, n, np.stack([clutter(w, fg, bg) for w in (2, 4, 8)]))
    print(f"{pol:14s} clutter (w 2/4/8): centre-4x4 mean", [round(float(z[12:16, 12:16].mean()), 2) for z in C], " max", [round(float(z.max()), 2) for z in C])
print("blank gray: max", round(float(zmap(b, n, np.full((1, 224, 224, 3), 128, np.uint8)).max()), 2))
