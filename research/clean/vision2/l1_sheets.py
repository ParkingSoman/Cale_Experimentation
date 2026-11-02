"""Pictures for L1: parent, hand child, learned neuron; learned neuron on child spots and on parent-only spots (held-out images)."""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw
from convnext import *
X, y = load_images(); names = [n.split(",")[0] for n in class_names()]; R = json.load(open(os.path.join(RES, "vision-l1-learn.json")))
CROP = {1: 72, 2: 136, 3: 224}; T = 88; k = 8
def crop(i, p, g):
    st = {28: 1, 14: 2, 7: 3}[g]; r, c = divmod(int(p), g); s = 224 / g; cs = CROP[st]; cy, cx = (r + .5) * s, (c + .5) * s
    t = int(np.clip(cy - cs / 2, 0, 224 - cs)); l = int(np.clip(cx - cs / 2, 0, 224 - cs)); return Image.fromarray(np.asarray(X[i])[t:t + cs, l:l + cs]).resize((T, T))
for key, o in R.items():
    d = np.load(os.path.join(HERE, "figs", "l1", f"acts_{key}.npz")); ev = d["ev"]; P = len(d["hp"]) // len(ev); g = int(round(P ** .5))
    hp, hs, hc = [a.reshape(len(ev), P) for a in (d["hp"], d["hs"])] + [d["hc"][:, 0].reshape(len(ev), P)]
    nz = lambda a: a / max(a.max(), 1e-6)
    def spots(sc):
        b = sc.argmax(1); v = sc[np.arange(len(sc)), b]; return [(i, b[i]) for i in np.argsort(-v)[:k]]
    po = nz(hp) - 3 * nz(np.maximum(hc, 0))
    rows = [(f"parent block {o['parent'][0]} n{o['parent'][1]}", spots(hp), hp), (f"child block {o['children'][0][0]} n{o['children'][0][1]} (hand-studied)", spots(hc), hc),
            ("LEARNED coarse neuron (replaces parent + children)", spots(hs), hs), ("learned neuron on the CHILD's strongest spots (% of its max)", spots(hc), hs),
            ("parent on the same child spots", spots(hc), hp), ("learned neuron on PARENT-ONLY spots", spots(po), hs), ("parent on the same parent-only spots", spots(po), hp)]
    W = T * k + 16; img = Image.new("RGB", (W, (T + 30) * len(rows) + 50), "white"); dr = ImageDraw.Draw(img)
    dr.text((8, 6), f"{key}: share of naive-cut damage won back {o['share_won_back']:.2f}; corr learned~parent {o['corr_learned_parent']:.2f}; partial corr with children given parent {o['partial_corr_children_given_parent']:.2f}", fill="black"); yy = 46
    for cap, sp, val in rows:
        dr.text((8, yy - 12), cap, fill="black"); mv = max(val.max(), 1e-6)
        for j, (i, q) in enumerate(sp): img.paste(crop(ev[i], q, g), (8 + j * T, yy)); dr.text((8 + j * T, yy + T), f"{names[y[ev[i]]][:9]} {100 * val[i, q] / mv:.0f}%", fill="darkgreen")
        yy += T + 30
    fp = os.path.join(HERE, "figs", "l1", f"sheet_{key}.png"); img.save(fp); print(fp)
