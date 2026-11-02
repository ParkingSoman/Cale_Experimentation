"""Hue tuning on natural images for b14 n33 and b14 n831: true HSV hue rotation (30 deg steps) of the top-20
images; response = median(image max after / before). Also flat full-field patches at 12 hues (S=0.7, V=0.7) + gray.
PREDICTIONS (before running):
 n33: peak at 0 deg (the original warm orange/red); rotations of +-90 deg or more drop below 0.4.
 n831: peak at 0; +-60 deg drops below 0.5; 180 deg (warm/olive -> blue) gives a negative image max.
 Flat patches: n831 > 30% of max for an olive/tan patch (hue 40-80), negative for blue (200-240). n33 flat orange
 patch NEGATIVE (large warm surfaces are its off pole)."""
from lib4 import *
import colorsys
X, y, ana = data(); A = acts()
def hue_rot(a, deg):
    from PIL import Image
    h = np.array(Image.fromarray(a).convert("HSV")); h[..., 0] = ((h[..., 0].astype(int) + int(round(deg / 360 * 256))) % 256).astype(np.uint8)
    return np.array(Image.fromarray(h, "HSV").convert("RGB"))
for b, n, MX in ((14, 33, 5.64), (14, 831, 5.79)):
    top = np.argsort(-A[f"max{b}"][:, n].astype(np.float32))[:20]; T = np.asarray(X[ana[top]]); base = zmap(b, n, T).max((1, 2))
    out = []
    for d in range(0, 360, 30):
        out.append(np.median(zmap(b, n, np.stack([hue_rot(t, d) for t in T])).max((1, 2)) / base))
    print(f"b{b} n{n} hue rotation 0..330 step 30, median max ratio:", [round(float(v), 2) for v in out])
    P = []
    for hh in range(0, 360, 30):
        r, g, bb = colorsys.hsv_to_rgb(hh / 360, .7, .7); P.append(np.full((224, 224, 3), [r * 255, g * 255, bb * 255], np.uint8))
    P.append(np.full((224, 224, 3), 128, np.uint8)); Z = zmap(b, n, np.stack(P))
    print(f"   flat patches hue 0..330 + gray, centre z %max:", [int(100 * z[7, 7] / MX) for z in Z])
