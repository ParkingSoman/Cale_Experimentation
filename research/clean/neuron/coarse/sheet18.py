"""N11 photo sheets: the top photos of a ResNet-18 stage-4 neuron (by pos, the max pre-ReLU input over positions),
on held + test photos. Usage: python sheet18.py TAG BLOCK NEURON [TAG BLOCK NEURON ...] -> one PNG per row group,
results/coarse/r18/sheets/<name>.png. Each row = one neuron: 10 top photos, each cropped (128 px) around
the neuron's strongest position when the capture holds it, class name under each."""
import os
import sys

import json

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "coarse", "r18")
CLEAN = os.path.join(HERE, "..", "..")
CI = json.load(open(os.path.join(HERE, "..", "results", "read", "imagenet_class_index.json")))
NAMES = [CI[str(i)][1] for i in range(1000)]
S = 112


class Photos:
    """numpy-only stand-in for net.Photos('read').raw (no torch on the Mac)."""

    def __init__(self):
        self.ix = np.load(os.path.join(CLEAN, "curves", "cache", "r18", "ix_train.npy"))
        self.X = np.load(os.path.join(CLEAN, "hf", "imagenet", "val_224.npy"), mmap_mode="r")

    def raw(self, val_ids):
        return np.stack([np.asarray(self.X[int(v)]) for v in val_ids])


def row(tag, b, j, ph, n=10):
    if tag == "S3":                                    # A's stage-3 output channel j (the shared parents); block arg ignored
        d = np.load(os.path.join(OUT, "read_s3.npz"))
        v = d["pos"][:, j].astype(np.float32)
        top = np.argsort(-v)[:n]
        raw = ph.raw(ph.ix[d["ix"][top]])
        img = Image.new("RGB", (n * S, S + 28), "white")
        dr = ImageDraw.Draw(img)
        for k in range(n):
            p = int(d["arg"][top[k], j])
            cx, cy = (p % 14 + 0.5) * 16, (p // 14 + 0.5) * 16
            l, t = int(min(max(cx - 48, 0), 128)), int(min(max(cy - 48, 0), 128))
            img.paste(Image.fromarray(raw[k]).crop((l, t, l + 96, t + 96)).resize((S, S)), (k * S, 14))
            dr.text((k * S + 2, S + 15), NAMES[d["lab"][top[k]]][:18], fill="black")
        dr.text((2, 0), f"A stage-3 channel {j}", fill="black")
        return img
    d = np.load(os.path.join(OUT, f"read_{tag}.npz"))
    v = d[f"pos{b}"][:, j].astype(np.float32)
    top = np.argsort(-v)[:n]
    raw = ph.raw(ph.ix[d["ix"][top]])
    img = Image.new("RGB", (n * S, S + 28), "white")
    dr = ImageDraw.Draw(img)
    for k in range(n):
        im = Image.fromarray(raw[k])
        if f"arg{b}" in d.files:                       # crop centred on the strongest position (7 x 7 grid, stride 32)
            p = int(d[f"arg{b}"][top[k], j])
            cx, cy = (p % 7 + 0.5) * 32, (p // 7 + 0.5) * 32
            l, t = int(min(max(cx - 64, 0), 96)), int(min(max(cy - 64, 0), 96))
            im = im.crop((l, t, l + 128, t + 128))
        img.paste(im.resize((S, S)), (k * S, 14))
        dr.text((k * S + 2, S + 15), NAMES[d["lab"][top[k]]][:18], fill="black")
    dr.text((2, 0), f"{tag} block {b} neuron {j}", fill="black")
    return img


def main():
    a = sys.argv[1:]
    ph = Photos()
    rows = [row(a[i], int(a[i + 1]), int(a[i + 2]), ph) for i in range(0, len(a), 3)]
    out = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    os.makedirs(os.path.join(OUT, "sheets"), exist_ok=True)
    name = "_".join(f"{a[i]}b{a[i + 1]}n{a[i + 2]}" for i in range(0, len(a), 3))[:120]
    f = os.path.join(OUT, "sheets", f"{name}.png")
    out.save(f)
    print(f)


if __name__ == "__main__":
    main()
