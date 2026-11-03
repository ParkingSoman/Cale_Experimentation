# serves: H-map
"""west round 2 drawings. Written BEFORE the probe run.
Predictions (can fail):
 Q1 s4b0i:167 fires on a drawn chain of linked rings (>=30% of its largest) more than on a row of the same rings not touching, and more than a solid straight bar.
 Q2 s3b1:130 fires on a vertical stem with a round knob on top more than on the same stem with a flat top or no knob.
 Q3 s4b0:35 fires on dark spots on a light animal-ish blob more than on the plain blob.
"""
import os
import numpy as np
from PIL import Image, ImageDraw

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "west_imgs")
os.makedirs(OUT, exist_ok=True)
S = 224
rng = np.random.default_rng(1)


def save(im, n):
    im.save(os.path.join(OUT, n + ".png"))


def base(col=(235, 235, 230)):
    im = Image.new("RGB", (S, S), col)
    return im, ImageDraw.Draw(im)


im, d = base()
pts = [(20 + i * 22, 112 + int(20 * np.sin(i / 1.5))) for i in range(10)]
for p in pts:
    d.ellipse([p[0] - 14, p[1] - 9, p[0] + 14, p[1] + 9], outline=(80, 80, 80), width=4)
save(im, "chain")
im, d = base()
for p in pts:
    d.ellipse([p[0] - 8, p[1] - 6, p[0] + 8, p[1] + 6], outline=(80, 80, 80), width=4)
save(im, "rings_apart")
im, d = base()
d.line(pts, fill=(80, 80, 80), width=8)
save(im, "bar")

im, d = base()
d.rectangle([104, 60, 120, 200], fill=(50, 50, 90)); d.ellipse([94, 30, 130, 66], fill=(200, 40, 40))
save(im, "stem_knob")
im, d = base()
d.rectangle([104, 60, 120, 200], fill=(50, 50, 90))
save(im, "stem_flat")
im, d = base()
d.rectangle([104, 60, 120, 200], fill=(50, 50, 90)); d.rectangle([94, 54, 130, 66], fill=(200, 40, 40))
save(im, "stem_cap_flat")

for nm, spots in (("blob_spots", True), ("blob_plain", False)):
    im, d = base((120, 160, 90))
    d.ellipse([40, 90, 180, 170], fill=(240, 240, 235)); d.ellipse([150, 60, 200, 110], fill=(240, 240, 235))
    if spots:
        for _ in range(35):
            x, y = int(rng.integers(55, 170)), int(rng.integers(100, 160)); r = int(rng.integers(3, 6))
            d.ellipse([x - r, y - r, x + r, y + r], fill=(30, 30, 30))
    save(im, nm)
