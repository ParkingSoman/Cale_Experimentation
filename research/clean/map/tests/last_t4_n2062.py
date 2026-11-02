"""16 n2062 vs 15 n3005 (strongest negative input) vs 15 n2065 (top input): same synthetic small-device drawings."""
import numpy as np
from PIL import Image, ImageDraw
from last_lib import *
BG, BOX, DK = (225, 225, 220), (60, 60, 65), (15, 15, 15)
NEU = [(16, 2062, 4.77), (15, 3005, 5.85), (15, 2065, 10.66), (15, 2584, 7.99)]


def dev(inner="none", w=(50, 70, 174, 160), rot=0):
    im = Image.new("RGB", (224, 224), BG); d = ImageDraw.Draw(im); x0, y0, x1, y1 = w
    if inner != "bare_circle": d.rounded_rectangle(w, radius=8, fill=BOX)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    if inner in ("circle", "bare_circle"): d.ellipse([cx - 32, cy - 32, cx + 32, cy + 32], fill=(150, 150, 155), outline=DK, width=4); d.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=DK)
    if inner == "dial":
        d.ellipse([cx - 36, cy - 36, cx + 36, cy + 36], fill=(200, 200, 200))
        for a in np.linspace(0, 2 * np.pi, 10, endpoint=False): px, py = cx + 24 * np.cos(a), cy + 24 * np.sin(a); d.ellipse([px - 6, py - 6, px + 6, py + 6], fill=DK)
    if inner == "grid":
        for i in range(4):
            for j in range(3): d.rectangle([x0 + 14 + j * 34, y0 + 10 + i * 20, x0 + 34 + j * 34, y0 + 22 + i * 20], fill=(190, 190, 190))
    if inner == "screen+grid":
        d.rectangle([x0 + 10, y0 + 8, x1 - 10, cy - 4], fill=(120, 170, 220))
        for i in range(2):
            for j in range(4): d.rectangle([x0 + 10 + j * 27, cy + 6 + i * 18, x0 + 30 + j * 27, cy + 18 + i * 18], fill=(190, 190, 190))
    if inner == "knobs+slot":
        d.rectangle([x0 + 12, y0 + 12, x1 - 12, y0 + 32], fill=DK)
        for j in range(3): d.ellipse([x0 + 16 + j * 34, y1 - 40, x0 + 40 + j * 34, y1 - 16], fill=(170, 170, 170), outline=DK, width=2)
    return np.array(im.rotate(rot, fillcolor=BG))


S = {"plain box": dev(), "box + round lens": dev("circle"), "round lens, no box": dev("bare_circle"), "box + dial ring": dev("dial"),
     "box + button grid": dev("grid"), "box + screen + keys": dev("screen+grid"), "box + slot + knobs": dev("knobs+slot"),
     "tall box + screen + keys (phone)": dev("screen+grid", w=(72, 30, 152, 194)), "box + lens rot90": dev("circle", rot=90),
     "blank": np.full((224, 224, 3), BG, np.uint8)}
A = np.stack(list(S.values())); R = {(b, n): zmaps(b, n, A).max((1, 2)) / mx_ for b, n, mx_ in NEU}
print(f"{'drawing':34s} " + " ".join(f"{b}n{n:<6d}" for b, n, _ in NEU) + "   (fraction of natural max)")
for i, k in enumerate(S): print(f"{k:34s} " + " ".join(f"{R[(b, n)][i]:8.2f}" for b, n, _ in NEU))
