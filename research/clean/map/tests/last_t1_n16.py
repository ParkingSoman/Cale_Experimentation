"""15 n16 test: thin upright arch (inverted U) vs other shapes, synthetic on gray. Prediction in last_predictions.md."""
import numpy as np
from PIL import Image, ImageDraw
from last_lib import zmaps
B, N, MAX = 15, 16, 10.6


def canvas(): im = Image.new("RGB", (224, 224), (150, 150, 150)); return im, ImageDraw.Draw(im)


def arch(w=4, size=120, rot=0, closed=False):
    im, d = canvas(); x0, y0 = 112 - size // 2, 112 - size // 2
    if closed: d.ellipse([x0, y0, x0 + size, y0 + size], outline=(30, 30, 30), width=w)
    else:
        d.arc([x0, y0, x0 + size, y0 + size], 180, 360, fill=(30, 30, 30), width=w)
        d.line([(x0 + w // 2, 112), (x0 + w // 2, y0 + size)], fill=(30, 30, 30), width=w); d.line([(x0 + size - w // 2, 112), (x0 + size - w // 2, y0 + size)], fill=(30, 30, 30), width=w)
    return np.array(im.rotate(rot, fillcolor=(150, 150, 150)))


def shapes():
    S = {}
    for w in (2, 4, 8, 16): S[f"arch w{w}"] = arch(w)
    for s in (60, 120, 180): S[f"arch size{s}"] = arch(4, s)
    S["arch rot90"] = arch(4, rot=90); S["arch rot180 (U)"] = arch(4, rot=180); S["ring"] = arch(4, closed=True)
    im, d = canvas(); d.line([(112, 40), (112, 184)], fill=(30, 30, 30), width=4); S["vertical line"] = np.array(im)
    im, d = canvas(); d.line([(40, 112), (184, 112)], fill=(30, 30, 30), width=4); S["horizontal line"] = np.array(im)
    im, d = canvas(); d.rectangle([82, 40, 142, 184], fill=(30, 30, 30)); S["filled tall rect"] = np.array(im)
    im, d = canvas(); d.ellipse([52, 52, 172, 172], fill=(30, 30, 30)); S["filled disk"] = np.array(im)
    # bucket: body + handle, body alone, handle alone
    def bucket(body=True, handle=True):
        im, d = canvas()
        if body: d.polygon([(70, 110), (154, 110), (144, 190), (80, 190)], fill=(90, 90, 100))
        if handle: d.arc([70, 40, 154, 180], 180, 360, fill=(30, 30, 30), width=4)
        return np.array(im)
    S["bucket body+handle"] = bucket(); S["bucket body only"] = bucket(handle=False); S["bucket handle only"] = bucket(body=False)
    S["blank gray"] = np.array(canvas()[0])
    for k in (3, 5): S[f"{k} legs (thin oblique lines)"] = None
    for k in (3, 5):
        im, d = canvas()
        for j in range(k): a = np.deg2rad(200 + j * 140 / (k - 1)); d.line([(112, 80), (112 + 110 * np.cos(a + np.pi), 80 - 110 * np.sin(a + np.pi))], fill=(30, 30, 30), width=3)
        S[f"{k} legs (thin oblique lines)"] = np.array(im)
    return S


S = shapes(); Z = zmaps(B, N, np.stack(list(S.values())))
for k, z in zip(S, Z): print(f"{k:32s} max {z.max():6.2f}  ({100*z.max()/MAX:5.0f}% of natural max)")
