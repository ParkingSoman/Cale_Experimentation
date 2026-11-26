"""Band sheets for the one-coarse triple (numpy + PIL, Mac). One PNG per neuron, 4 rows of 10 crops:
  top     : ranks 0-9 by peak (max over positions of the input before the ReLU), crop centred on the peak
  strong  : 10 random photos from ranks 100-400, crop on the peak
  weak    : 10 random photos from ranks 1000-3000 (weak but positive), crop on the peak
  NEG rows : the same three bands ranked by the MIN over positions, crop centred on the min (reading rule 1)
Crop: 112 px around the peak (stage 4: 7 x 7 grid, 32 px cells; stage 3: 14 x 14 grid, 16 px cells).
Usage: python3 oc_bands.py  -> results/bands_<name>.png"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
R18 = os.path.join(HERE, "..", "..", "neuron", "results", "coarse", "r18")
CLEAN = os.path.join(HERE, "..", "..")
OUT = os.path.join(HERE, "..", "results")
CI = json.load(open(os.path.join(CLEAN, "neuron", "results", "read", "imagenet_class_index.json")))
NAMES = [CI[str(i)][1] for i in range(1000)]
IXT = np.load(os.path.join(CLEAN, "curves", "cache", "r18", "ix_train.npy"))
X = np.load(os.path.join(CLEAN, "hf", "imagenet", "val_224.npy"), mmap_mode="r")
S, C = 104, int(os.environ.get("OC_CROP", 112))
NEURONS = [("parent_s3_69", "read_s3.npz", "pos", "arg", "mean", 69, 14),
           ("A_n305", "read_A.npz", "pos0", "arg0", "mean0", 305, 7),
           ("A_n97", "read_A.npz", "pos0", "arg0", "mean0", 97, 7),
           ("A_n23", "read_A.npz", "pos0", "arg0", "mean0", 23, 7),
           ("A_n297", "read_A.npz", "pos0", "arg0", "mean0", 297, 7),
           ("A_n108", "read_A.npz", "pos0", "arg0", "mean0", 108, 7),
           ("A_n163", "read_A.npz", "pos0", "arg0", "mean0", 163, 7),
           ("B1_n101", "read_B128aug_s1.npz", "pos0", "arg0", "mean0", 101, 7)]


def crop(img, p, grid):
    cell = 224 // grid
    cx, cy = (p % grid + 0.5) * cell, (p // grid + 0.5) * cell
    l, t = int(min(max(cx - C / 2, 0), 224 - C)), int(min(max(cy - C / 2, 0), 224 - C))
    out = img.crop((l, t, l + C, t + C))
    dr = ImageDraw.Draw(out)                     # red box = the exact grid cell of the max (or min)
    dr.rectangle((cx - cell / 2 - l, cy - cell / 2 - t, cx + cell / 2 - l - 1, cy + cell / 2 - t - 1), outline=(255, 0, 0), width=2)
    return out


NEG = None


def bands(v, seed):
    o = np.argsort(-v)
    rng = np.random.default_rng(seed)
    return [o[:10], np.sort(rng.choice(o[100:400], 10, replace=False)), np.sort(rng.choice(o[1000:3000], 10, replace=False))]


def sheet(name, f, kp, ka, km, j, grid):
    """Rule 1: both poles with the same tool. Positive: max position crops. Negative: min position crops (oc_neg.npz)."""
    global NEG
    NEG = NEG if NEG is not None else np.load(os.path.join(OUT, "oc_neg.npz"))
    d = NEG if name.startswith("parent") else np.load(os.path.join(R18, f))
    assert (d["ix"] == NEG["ix"]).all()
    if name.startswith("parent"):
        v, arg = NEG["p_pos"].astype(np.float32), NEG["p_arg"]
        nv, narg = NEG["p_neg"].astype(np.float32), NEG["p_nar"]
    else:
        v, arg = d[kp][:, j].astype(np.float32), d[ka][:, j]
        if name.startswith("B1"):
            nv, narg = NEG["c_neg"][:, 0].astype(np.float32), NEG["c_nar"][:, 0]
        else:
            q = list(NEG["kids"]).index(j)
            nv, narg = NEG["k_neg"][:, q].astype(np.float32), NEG["k_nar"][:, q]
    P, N = bands(v, j), bands(-nv, j + 1)
    rows = [(f"POS {b}", i, arg) for b, i in zip(("top", "strong 100-400", "weak 1000-3000"), P)]
    rows += [(f"NEG {b}", i, narg) for b, i in zip(("top", "strong 100-400", "weak 1000-3000"), N)]
    img = Image.new("RGB", (10 * S, len(rows) * (S + 30)), "white")
    dr = ImageDraw.Draw(img)
    for r, (lab, idx, aa) in enumerate(rows):
        y = r * (S + 30)
        dr.text((2, y), f"{name}  {lab}", fill="black")
        for k, n in enumerate(idx):
            im = Image.fromarray(np.asarray(X[int(IXT[d["ix"][n]])]))
            im = crop(im, int(aa[n]), grid)
            img.paste(im.resize((S, S)), (k * S, y + 13))
            dr.text((k * S + 2, y + S + 14), NAMES[d["lab"][n]][:17], fill="black")
    p = os.path.join(OUT, f"bands_{name}{os.environ.get('OC_SUFFIX', '')}.png")
    img.save(p)
    print(p)


if __name__ == "__main__":
    want = sys.argv[1:]
    for nr in NEURONS:
        if not want or nr[0] in want:
            sheet(*nr)
