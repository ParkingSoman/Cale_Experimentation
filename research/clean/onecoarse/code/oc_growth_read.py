"""Read oc_growth.npz (Mac): one row of top-10 crops per checkpoint (box on the cell), and per checkpoint the
correlation of n101's per-photo max with the parent's peak and with each child's (oc_step1.npz, same photo order)."""
import json
import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "..")
OUT = os.path.join(HERE, "..", "results")
G = np.load(os.path.join(OUT, "oc_growth.npz"))
S1 = np.load(os.path.join(OUT, "oc_step1.npz"))
assert (G["ix"] == S1["ix"]).all()
CI = json.load(open(os.path.join(CLEAN, "neuron", "results", "read", "imagenet_class_index.json")))
NAMES = [CI[str(i)][1] for i in range(1000)]
IXT = np.load(os.path.join(CLEAN, "curves", "cache", "r18", "ix_train.npy"))
X = np.load(os.path.join(CLEAN, "hf", "imagenet", "val_224.npy"), mmap_mode="r")
STEPS = [1, 3, 10, 30, 100, 300, 1000, 3000, 6250]
KIDS = [305, 97, 23, 297, 108, 163]
S, C = 104, 112
z = lambda v: (v - v.mean()) / (v.std() + 1e-6)
ppk = S1["ppk"].astype(np.float32)
Apos = S1["A_pos"].astype(np.float32)
print("step | corr parent | corr children " + " ".join(f"n{j}" for j in KIDS) + " | top-200 classes")
img = Image.new("RGB", (10 * S, len(STEPS) * (S + 28)), "white")
dr = ImageDraw.Draw(img)
for i, st in enumerate(STEPS):
    v = G[f"pos_{st}"].astype(np.float32)
    cp = float((z(v) * z(ppk)).mean())
    ck = [float((z(v) * z(Apos[:, j])).mean()) for j in KIDS]
    top = np.argsort(-v)
    u, c = np.unique(S1["lab"][top[:200]], return_counts=True)
    cl = [NAMES[x] for x in u[np.argsort(-c)][:6]]
    print(f"{st:5d} | {cp:+.2f} | " + " ".join(f"{x:+.2f}" for x in ck) + f" | {cl}")
    y = i * (S + 28)
    dr.text((2, y), f"B1 n101 at step {st}: corr parent {cp:+.2f}; children " + ", ".join(f"n{j} {x:+.2f}" for j, x in zip(KIDS, ck)), fill="black")
    for k, n in enumerate(top[:10]):
        im = Image.fromarray(np.asarray(X[int(IXT[G["ix"][n]])]))
        p = int(G[f"arg_{st}"][n]); cx, cy = (p % 7 + .5) * 32, (p // 7 + .5) * 32
        l, t = int(min(max(cx - C / 2, 0), 224 - C)), int(min(max(cy - C / 2, 0), 224 - C))
        im = im.crop((l, t, l + C, t + C))
        ImageDraw.Draw(im).rectangle((cx - 16 - l, cy - 16 - t, cx + 15 - l, cy + 15 - t), outline=(255, 0, 0), width=2)
        img.paste(im.resize((S, S)), (k * S, y + 13))
        dr.text((k * S + 2, y + S + 14), NAMES[S1["lab"][n]][:17], fill="black")
img.save(os.path.join(OUT, "growth_n101.png"))
print(os.path.join(OUT, "growth_n101.png"))
