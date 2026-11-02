"""b14 n831 negative pole: blue sky / water / cool scenes. Is the off pole the opposite HUE?
PREDICTION (before running): rotating the hue of the 24 most-negative images by 180 deg (blue -> yellow-orange)
raises their median image max by > +1.5 and median mean z by > +1.0; graying them raises mean z (off pole needs colour)."""
from lib4 import *
from PIL import Image
b, n = 14, 831; X, y, ana = data(); A = acts()
def hr(a, d):
    h = np.array(Image.fromarray(a).convert("HSV")); h[..., 0] = ((h[..., 0].astype(int) + int(d / 360 * 256)) % 256).astype(np.uint8); return np.array(Image.fromarray(h, "HSV").convert("RGB"))
o = np.argsort(A[f"min{b}"][:, n].astype(np.float32))[:24]; T = np.asarray(X[ana[o]])
for lab, I in (("original", T), ("hue+180", np.stack([hr(t, 180) for t in T])), ("hue+120", np.stack([hr(t, 120) for t in T])), ("hue+240", np.stack([hr(t, 240) for t in T])), ("gray", np.stack([np.array(Image.fromarray(t).convert('L').convert('RGB')) for t in T]))):
    Z = zmap(b, n, I); print(f"{lab:9s} median max {np.median(Z.max((1,2))):5.2f}  median mean {np.median(Z.mean((1,2))):5.2f}")
