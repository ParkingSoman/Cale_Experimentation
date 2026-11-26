"""Step-2 sheets from oc_split.npz (Mac, numpy + PIL). One PNG per neuron, 6 rows x 10 tiles:
POS top / strong / weak (ranked by the max of value - offset, box on the max cell) and NEG top / strong / weak
(ranked by the INPUT-ONLY min, value - offset, box on that cell). Each tile = the whole photo with the map of
(value - offset) laid over it: red = inputs push up, blue = inputs push down (each side on its own scale).
Usage: python3 oc_sheet2.py [p k305 k97 k23 k297 k108 k163 c101]"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "..")
OUT = os.path.join(HERE, "..", "results")
D = np.load(os.path.join(OUT, "oc_split.npz"))
CI = json.load(open(os.path.join(CLEAN, "neuron", "results", "read", "imagenet_class_index.json")))
NAMES = [CI[str(i)][1] for i in range(1000)]
IXT = np.load(os.path.join(CLEAN, "curves", "cache", "r18", "ix_train.npy"))
X = np.load(os.path.join(CLEAN, "hf", "imagenet", "val_224.npy"), mmap_mode="r")
ROW = {int(r): i for i, r in enumerate(D["rows"])}
S = 150
ZOOM = int(os.environ.get("OC_ZOOM", 0))
TITLE = {"p": "parent s3:69", "c101": "coarse B1 n101"}


def tile(n, r, cell, scale):
    m = D[f"map_{n}"][ROW[int(r)]] - float(D[f"{n}_off"])
    g = m.shape[0]
    a = np.where(m > 0, m / scale[0], m / scale[1])
    a = np.asarray(Image.fromarray(np.clip(a, -1, 1).astype(np.float32)).resize((S, S), Image.NEAREST))
    raw = np.asarray(X[int(IXT[D["ix"][r]])])
    base = np.asarray(Image.fromarray(raw).resize((S, S))).astype(np.float32) * 0.55
    over = np.zeros_like(base); over[..., 0] = np.clip(a, 0, 1) * 255; over[..., 2] = np.clip(-a, 0, 1) * 255
    w = np.abs(a)[..., None] * 0.7
    im = Image.fromarray(np.clip(base * (1 - w) + over * w + base * w * 0.3, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    c = S / g; rr, cc = divmod(int(cell), g)
    dr.rectangle((cc * c, rr * c, (cc + 1) * c - 1, (rr + 1) * c - 1), outline=(255, 255, 0), width=2)
    if ZOOM:                                   # crop ZOOM photo pixels around the cell, from the RAW photo (no overlay)
        cx, cy = (cc + .5) * 224 / g, (rr + .5) * 224 / g
        l, t = int(min(max(cx - ZOOM / 2, 0), 224 - ZOOM)), int(min(max(cy - ZOOM / 2, 0), 224 - ZOOM))
        im = Image.fromarray(raw).crop((l, t, l + ZOOM, t + ZOOM)).resize((S, S))
        k = S / ZOOM; cs = 224 / g
        ImageDraw.Draw(im).rectangle(((cx - cs / 2 - l) * k, (cy - cs / 2 - t) * k, (cx + cs / 2 - l) * k - 1, (cy + cs / 2 - t) * k - 1), outline=(255, 255, 0), width=2)
    return im


def sheet(n):
    P, N = D[f"{n}_bpos"], D[f"{n}_bneg"]
    allm = np.stack([D[f"map_{n}"][ROW[int(r)]] for r in np.concatenate([P, N])]) - float(D[f"{n}_off"])
    scale = (np.quantile(allm[allm > 0], .999), np.quantile(-allm[allm < 0], .999))
    rows = [("POS " + b, P[k * 10:(k + 1) * 10], D[f"{n}_amax"]) for k, b in enumerate(("top", "strong 100-400", "weak 1000-3000"))]
    rows += [("NEG (input only) " + b, N[k * 10:(k + 1) * 10], D[f"{n}_amin"]) for k, b in enumerate(("top", "strong 100-400", "weak 1000-3000"))]
    img = Image.new("RGB", (10 * S, len(rows) * (S + 28)), "white")
    dr = ImageDraw.Draw(img)
    for i, (lab, idx, am) in enumerate(rows):
        y = i * (S + 28)
        dr.text((2, y), f"{TITLE.get(n, 'A b0 n' + n[1:])}  {lab}  (offset {float(D[n + '_off']):+.2f}; red up {scale[0]:.2f}, blue down {scale[1]:.2f}; yellow box = the cell)", fill="black")
        for k, r in enumerate(idx):
            img.paste(tile(n, r, am[r], scale), (k * S, y + 13))
            dr.text((k * S + 2, y + S + 14), NAMES[D["lab"][r]][:22], fill="black")
    p = os.path.join(OUT, f"s2_{n}{'_zoom' if ZOOM else ''}.png")
    img.save(p)
    print(p)


if __name__ == "__main__":
    for n in (sys.argv[1:] or ["p", "k305", "k97", "k23", "k297", "k108", "k163", "c101"]):
        sheet(n)
