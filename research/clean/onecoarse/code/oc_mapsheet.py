"""Map sheets: the same 6 band rows as oc_bands.py, but each tile is the WHOLE photo with the neuron's map laid over it
(red = input before the ReLU above 0, scaled by its 99.9th percentile; blue = below 0, scaled by ITS 99.9th
percentile, so a weak negative side stays visible). One tile shows where the neuron goes up and where it goes down in the same photo.
Usage: python3 oc_mapsheet.py [neuron names]  -> results/maps_<name>.png"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ob", os.path.join(HERE, "oc_bands.py"))
ob = importlib.util.module_from_spec(spec); spec.loader.exec_module(ob)
M = np.load(os.path.join(ob.OUT, "oc_maps.npz"))
NEG = np.load(os.path.join(ob.OUT, "oc_neg.npz"))
ROW = {int(r): i for i, r in enumerate(M["rows"])}
S = 150


def mp(name, j, n):
    i = ROW[int(n)]
    if name.startswith("parent"):
        return M["pz"][i]
    if name.startswith("B1"):
        return M["c"][i]
    return M["kids"][i, list(NEG["kids"]).index(j)]


def tile(raw, m, scale):
    im = Image.fromarray(raw).resize((S, S)).convert("RGB")
    sp, sn = scale
    a = np.where(m > 0, m / sp, m / sn)                 # red scaled by the positive scale, blue by the negative scale
    a = np.asarray(Image.fromarray(np.clip(a, -1, 1).astype(np.float32)).resize((S, S), Image.NEAREST))
    base = np.asarray(im).astype(np.float32) * 0.55
    over = np.zeros_like(base)
    over[..., 0] = np.clip(a, 0, 1) * 255
    over[..., 2] = np.clip(-a, 0, 1) * 255
    w = np.abs(a)[..., None] * 0.7
    return Image.fromarray(np.clip(base * (1 - w) + over * w + base * w * 0.3, 0, 255).astype(np.uint8))


def sheet(name, f, kp, ka, km, j, grid):
    d = NEG if name.startswith("parent") else np.load(os.path.join(ob.R18, f))
    if name.startswith("parent"):
        v, nv = NEG["p_pos"], NEG["p_neg"]
    elif name.startswith("B1"):
        v, nv = d[kp][:, j], NEG["c_neg"][:, 0]
    else:
        v, nv = d[kp][:, j], NEG["k_neg"][:, list(NEG["kids"]).index(j)]
    P, N = ob.bands(v.astype(np.float32), j), ob.bands(-nv.astype(np.float32), j + 1)
    rows = [(f"POS {b}", i) for b, i in zip(("top", "strong", "weak"), P)] + [(f"NEG {b}", i) for b, i in zip(("top", "strong", "weak"), N)]
    allm = np.stack([mp(name, j, n) for _, idx in rows for n in idx])
    scale = (np.quantile(allm[allm > 0], 0.999), np.quantile(-allm[allm < 0], 0.999))
    img = Image.new("RGB", (10 * S, len(rows) * (S + 28)), "white")
    dr = ImageDraw.Draw(img)
    for r, (lab, idx) in enumerate(rows):
        y = r * (S + 28)
        dr.text((2, y), f"{name}  {lab}  (red = up, scale {scale[0]:.2f}; blue = down, scale {scale[1]:.2f})", fill="black")
        for k, n in enumerate(idx):
            raw = np.asarray(ob.X[int(ob.IXT[NEG["ix"][n]])])
            img.paste(tile(raw, mp(name, j, n), scale), (k * S, y + 13))
            dr.text((k * S + 2, y + S + 14), ob.NAMES[NEG["lab"][n]][:22], fill="black")
    p = os.path.join(ob.OUT, f"maps_{name}.png")
    img.save(p)
    print(p)


if __name__ == "__main__":
    want = sys.argv[1:]
    for nr in ob.NEURONS:
        if not want or nr[0] in want:
            sheet(*nr)
