"""16 n384 test: tread band (row of wheels in a rounded band) vs parts; plus occlusion on real half-track/tank images."""
import numpy as np
from PIL import Image, ImageDraw
from last_lib import *
B, N, MAX = 16, 384, 12.0
BG, BODY, DARK = (170, 170, 160), (110, 105, 80), (40, 40, 40)


def draw(body=True, band=True, wheels=5, rot=0, wheel_r=12, two_wheels=False):
    im = Image.new("RGB", (224, 224), BG); d = ImageDraw.Draw(im)
    if body: d.polygon([(30, 130), (50, 80), (180, 80), (200, 130)], fill=BODY)
    if band: d.rounded_rectangle([25, 128, 205, 172], radius=22, fill=DARK)
    if wheels:
        xs = np.linspace(48, 182, wheels)
        for x in xs: d.ellipse([x - wheel_r, 150 - wheel_r, x + wheel_r, 150 + wheel_r], fill=(120, 120, 115), outline=(20, 20, 20), width=2)
    if two_wheels:
        for x in (60, 170): d.ellipse([x - 20, 130, x + 20, 170], fill=DARK)
    return np.array(im.rotate(rot, fillcolor=BG))


S = {"full: body+band+5 wheels": draw(), "body only": draw(band=False, wheels=0), "band+wheels only": draw(body=False),
     "band only (no wheels)": draw(body=False, wheels=0), "wheels only (no band)": draw(body=False, band=False),
     "body+band no wheels": draw(wheels=0), "full rot90": draw(rot=90), "full rot180": draw(rot=180),
     "car-like: body + 2 wheels": draw(band=False, wheels=0, two_wheels=True), "full, 8 small wheels": draw(wheels=8, wheel_r=8),
     "blank": np.full((224, 224, 3), BG, np.uint8)}
Z = zmaps(B, N, np.stack(list(S.values())))
for k, z in zip(S, Z): print(f"{k:30s} max {z.max():6.2f} ({100*z.max()/MAX:4.0f}% of max)")
# occlusion on real images: half-track and tank analysis images, gray out bottom half / top half
X, y, ana = data(); nm = names(); ya = y[ana]
for c in ("half track", "tank"):
    ix = ana[ya == nm.index(c)]; im = np.stack([np.asarray(X[i]) for i in ix]); base = zmaps(B, N, im).max((1, 2))
    res = {}
    for part, sl in (("bottom third", slice(150, 224)), ("top third", slice(0, 75)), ("middle third", slice(75, 150))):
        e = im.copy(); e[:, sl] = 128; res[part] = zmaps(B, N, e).max((1, 2))
    ok = base > 2
    print(c, f"n={ok.sum()}/{len(ok)} with max>2, median base {np.median(base[ok]):.2f};", ", ".join(f"{p} grayed -> {np.median(r[ok]/base[ok]):.2f}" for p, r in res.items()))
