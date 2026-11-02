"""b5 n135 follow-up: fine angle sweep near 15 deg, segment length (end-stopping), circles/arcs (curvature),
stripes of segments, polarity."""
import json, numpy as np
from PIL import Image, ImageDraw
from early_lib import *

def seg(angle, length, width=3, fg=0, bg=200):
    return line_img(angle, width, fg, bg, length=length)

def circle(r, width=3, fg=0, bg=200, filled=False):
    im = Image.new("RGB", (224, 224), (bg,) * 3); d = ImageDraw.Draw(im)
    if filled: d.ellipse([112 - r, 112 - r, 112 + r, 112 + r], fill=(fg,) * 3)
    else: d.ellipse([112 - r, 112 - r, 112 + r, 112 + r], outline=(fg,) * 3, width=width)
    return np.array(im)

def parallel(angle, n, gap, length, width=3, fg=0, bg=200):
    im = Image.new("RGB", (224, 224), (bg,) * 3); d = ImageDraw.Draw(im); a = np.deg2rad(angle)
    ux, uy = np.cos(a), -np.sin(a); px, py = -uy, ux
    for k in range(n):
        o = (k - (n - 1) / 2) * gap; cx, cy = 112 + px * o, 112 + py * o
        d.line([(cx - ux * length / 2, cy - uy * length / 2), (cx + ux * length / 2, cy + uy * length / 2)], fill=(fg,) * 3, width=width)
    return np.array(im)

stim, lab = [], []
for a in range(-10, 45, 5):
    for L in (8, 16, 32, 64, 300):
        for pol in ("dark", "light"):
            fg, bg = (0, 200) if pol == "dark" else (200, 0); stim.append(seg(a, L, 3, fg, bg)); lab.append(("seg", a, L, pol))
for r in (6, 12, 24, 48, 96):
    for pol in ("dark", "light"):
        fg, bg = (0, 200) if pol == "dark" else (200, 0)
        stim.append(circle(r, 3, fg, bg)); lab.append(("ring", 0, r, pol)); stim.append(circle(r, 3, fg, bg, True)); lab.append(("disc", 0, r, pol))
for a in (15, 165, 90, 0):
    for n_, gap in ((3, 8), (3, 16), (6, 8)):
        for L in (24, 64):
            stim.append(parallel(a, n_, gap, L)); lab.append((f"par{n_}x{gap}", a, L, "dark"))
r = stat(multi([(5, 135)], np.stack(stim))[(5, 135)]) / 3.67
print("5,135 segments: rows angle, cols length 8/16/32/64/inf; dark | light")
for a in range(-10, 45, 5):
    d = [r[i] for i, l in enumerate(lab) if l[0] == "seg" and l[1] == a and l[3] == "dark"]
    li = [r[i] for i, l in enumerate(lab) if l[0] == "seg" and l[1] == a and l[3] == "light"]
    print(f"  {a:3d}", " ".join(f"{x:5.2f}" for x in d), " | ", " ".join(f"{x:5.2f}" for x in li))
for i, l in enumerate(lab):
    if l[0] != "seg": print("  ", l, f"{r[i]:.2f}")
json.dump({"labels": lab, "resp": r.tolist()}, open(__file__.replace(".py", ".json"), "w"))
