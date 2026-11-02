"""0 n346 follow-up: flat fields give ~0 at block 0. Use tinted texture (gray noise blurred 1 px, times hue colour)
and hue-rotated copies of 831's top-20 images. Report mean z over the image.
PREDICTION: n346 higher for purple (270-300) than for green (90-120)."""
from lib4 import *
import colorsys
from PIL import Image, ImageFilter
rng = np.random.default_rng(0); X, y, ana = data(); A = acts()
tex = np.array(Image.fromarray(rng.integers(60, 255, (224, 224)).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1))).astype(np.float32) / 255
P = []
for hh in range(0, 360, 30):
    c = np.array(colorsys.hsv_to_rgb(hh / 360, .7, 1.)) * 255; P.append((tex[..., None] * c).astype(np.uint8))
P = np.stack(P)
def hr(a, d):
    h = np.array(Image.fromarray(a).convert("HSV")); h[..., 0] = ((h[..., 0].astype(int) + int(d / 360 * 256)) % 256).astype(np.uint8); return np.array(Image.fromarray(h, "HSV").convert("RGB"))
top = np.argsort(-A["max14"][:, 831].astype(np.float32))[:20]; T = np.asarray(X[ana[top]])
for b, n in ((0, 346), (0, 58), (0, 24)):
    print(f"b{b} n{n} tinted texture hue 0..330 mean z:", [round(float(z.mean()), 1) for z in zmap(b, n, P)])
    print(f"   831-top images hue-rot 0..330 mean z:", [round(float(zmap(b, n, np.stack([hr(t, d) for t in T])).mean()), 2) for d in range(0, 360, 30)])
