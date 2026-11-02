"""b12 n136: is the PLAIN GROUND part of the job? (lines alone reach only 35-49%; inverted strokes go negative.)
PREDICTION (before running): (1) Adding fine texture to the ground kills it: the V stroke on a light ground with
pixel noise (sd 30) drops below 10% of max; on the top-24 natural images, the same noise drops the peak to <50% of
before (median). (2) Smoothing the ground helps or keeps: a 3x3 median filter keeps >80%."""
from lib4 import *
from PIL import Image, ImageDraw, ImageFilter
b, n = 12, 136; MX = 2.83; rng = np.random.default_rng(0); X, y, ana = data(); A = acts()
def V(bg_noise=0, fine=False):
    P = Image.new("RGB", (224, 224), (190, 190, 190)); D = ImageDraw.Draw(P)
    if fine:
        for _ in range(400):
            x0, y0 = rng.uniform(0, 224, 2); D.line([(x0, y0), (x0 + rng.uniform(-8, 8), y0 + rng.uniform(-8, 8))], fill=(150, 150, 150), width=1)
    D.line([(80, 30), (112, 170), (144, 30)], fill=(40, 40, 40), width=6); a = np.array(P).astype(np.float32)
    return np.clip(a + rng.normal(0, bg_noise, a.shape), 0, 255).astype(np.uint8)
I = np.stack([V(), V(15), V(30), V(fine=True)]); Z = zmap(b, n, I)
print("V stroke: plain / noise15 / noise30 / fine-scratch ground  %max:", [int(100 * z.max() / MX) for z in Z])
v = A[f"max{b}"][:, n].astype(np.float32); top = np.argsort(-v)[:24]; T = np.asarray(X[ana[top]]); base = zmap(b, n, T).max((1, 2))
for name, f in (("noise sd15", lambda a: np.clip(a + rng.normal(0, 15, a.shape), 0, 255)), ("noise sd30", lambda a: np.clip(a + rng.normal(0, 30, a.shape), 0, 255)),
                ("median3", lambda a: np.array(Image.fromarray(a).filter(ImageFilter.MedianFilter(3)))), ("median5", lambda a: np.array(Image.fromarray(a).filter(ImageFilter.MedianFilter(5))))):
    E = np.stack([f(t.astype(np.float32) if "noise" in name else t) for t in T]).astype(np.uint8)
    print(f"top-24 {name}: median ratio {np.median(zmap(b, n, E).max((1, 2)) / base):.2f}")
