"""0 n346: is it a colour-EDGE opponent? Half-plane images, purple|green and green|purple, 4 orientations; also
yellow-olive|blue. Report max and min of z (block 0, 56x56). Also max/min on 831-top images hue-rotated.
PREDICTION: purple/green boundaries give |z| > 5 with sign set by which side is purple."""
from lib4 import *
from PIL import Image
X, y, ana = data(); A = acts()
PU, GR, YO, BL = (150, 60, 200), (60, 170, 60), (170, 160, 50), (50, 90, 200)
def hp(c1, c2, orient):
    a = np.zeros((224, 224, 3), np.uint8); a[:] = c1
    if orient == 0: a[:, 112:] = c2
    elif orient == 1: a[112:, :] = c2
    elif orient == 2: a[:, :112] = c2
    else: a[:112, :] = c2
    return a
for lab, c1, c2 in (("purple|green", PU, GR), ("yellowolive|blue", YO, BL), ("green|gray", GR, (128, 128, 128)), ("purple|gray", PU, (128, 128, 128))):
    Z = zmap(0, 346, np.stack([hp(c1, c2, o) for o in range(4)]))
    print(f"{lab:18s} max per orientation {[round(float(z.max()),1) for z in Z]}  min {[round(float(z.min()),1) for z in Z]}")
def hr(a, d):
    h = np.array(Image.fromarray(a).convert("HSV")); h[..., 0] = ((h[..., 0].astype(int) + int(d / 360 * 256)) % 256).astype(np.uint8); return np.array(Image.fromarray(h, "HSV").convert("RGB"))
top = np.argsort(-A["max14"][:, 831].astype(np.float32))[:20]; T = np.asarray(X[ana[top]]); Z0 = zmap(0, 346, T)
print("831-top: 346 median max", round(float(np.median(Z0.max((1,2)))),1), "median min", round(float(np.median(Z0.min((1,2)))),1), "frac positions z>2:", round(float((Z0>2).mean()),3), "z<-2:", round(float((Z0<-2).mean()),3))
r = np.asarray(X[ana[np.random.default_rng(0).choice(len(ana), 200, replace=False)]]); Zr = zmap(0, 346, r)
print("random: frac z>2:", round(float((Zr>2).mean()),3), "z<-2:", round(float((Zr<-2).mean()),3))
