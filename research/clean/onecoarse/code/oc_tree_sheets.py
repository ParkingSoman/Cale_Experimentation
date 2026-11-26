"""CALC 3.7 reader sheets (checklist steps 1-4). From results/oc_tree.npz. Photos: the 4,000 test photos of oc_tree.py.
Units: 11 R1 children of s3:69 (pos pole, and NEG pole INSIDE the parent's group = parent-on cells only), the parent
(pos only: its value at a cell = max of s3:69 over the cell's 3 x 3 stage-3 window; s3:69 is after a ReLU, so its
negative side is not in this file), and B's n101 (pos, and neg inside the parent's group).
Tile = 140 x 140 crop centred on the stage-4 cell (stride 32), shown at 128, above the whole photo with the crop boxed.
Bands: TOP ranks 1-12, STRONG 8 of ranks 13-80, WEAK 6 of ranks 401-800 (one cell per photo).
Codes are anonymous; key in results/tree_cards/KEY.json (readers never open it).
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(CLEAN, "curves", "map_agents", "resnet18"))
import net  # noqa: E402

RES = os.path.join(HERE, "..", "results")
OUTD = os.path.join(RES, "tree_cards")
KIDS = [305, 97, 23, 163, 385, 297, 464, 237, 108, 510, 275]
T, HALF = 128, 70


def crop(img, cell):
    r, c = divmod(int(cell), 7)
    cy, cx = r * 32 + 16 + HALF, c * 32 + 16 + HALF
    p = np.pad(img, ((HALF, HALF), (HALF, HALF), (0, 0)), constant_values=128)
    return p[cy - HALF:cy + HALF, cx - HALF:cx + HALF]


def tile(img, cell):
    t = Image.new("RGB", (T, T + T // 2 + 4), "white")
    t.paste(Image.fromarray(crop(img, cell)).resize((T, T)), (0, 0))
    ctx = Image.fromarray(img).resize((T // 2, T // 2)); d = ImageDraw.Draw(ctx)
    r, c = divmod(int(cell), 7); s = (T / 2) / 224
    d.rectangle([(c * 32 + 16 - HALF) * s, (r * 32 + 16 - HALF) * s, (c * 32 + 16 + HALF) * s, (r * 32 + 16 + HALF) * s], outline="red", width=1)
    t.paste(ctx, (0, T + 4))
    return t


def sheet(ph, ids, rows, cells, path):
    pick = list(range(12)) + list(np.linspace(12, 79, 8).astype(int)) + list(np.linspace(400, 799, 6).astype(int))
    imgs = ph.raw(ids[rows[:40]])
    W = np.stack([crop(i, c).astype(np.float32) for i, c in zip(imgs, cells[:40])])
    g = W.mean(3); gy, gx = np.gradient(g, axis=(1, 2)); e = np.hypot(gx, gy).mean(0); e = 255 * e / (e.max() + 1e-6)
    th = T + T // 2 + 4
    lines = [("MEAN of top 40 crops: colour, edges", None), ("TOP band, ranks 1-6", pick[0:6]), ("TOP band, ranks 7-12", pick[6:12]),
             ("STRONG band (ranks 13-80)", pick[12:20]), ("WEAK band (ranks 401-800)", pick[20:26])]
    im = Image.new("RGB", (8 * (T + 4), sum(T + 20 if s is None else th + 20 for _, s in lines)), "white"); d = ImageDraw.Draw(im)
    y = 0
    for lab, sel in lines:
        d.text((4, y + 4), lab, fill="black"); y += 18
        if sel is None:
            im.paste(Image.fromarray(W.mean(0).clip(0, 255).astype(np.uint8)).resize((T, T)), (0, y))
            im.paste(Image.fromarray(e.astype(np.uint8)).resize((T, T)).convert("RGB"), (T + 4, y)); y += T + 2; continue
        for k, (img, j) in enumerate(zip(ph.raw(ids[rows[sel]]), sel)):
            im.paste(tile(img, cells[j]), (k * (T + 4), y))
        y += th + 2
    im.save(path)


def ranked_pos(X):  # X (N, 49): one cell per photo, by the max
    o = np.argsort(-X.max(1)); return o, X[o].argmax(1)


def ranked_neg_in(X, on):  # most negative parent-on cell per photo, photos ranked by it
    Y = np.where(on, X, np.inf); m = Y.min(1); ok = np.isfinite(m)
    o = np.where(ok)[0][np.argsort(m[ok])]; return o, Y[o].argmin(1)


def main():
    D = np.load(os.path.join(RES, "oc_tree.npz"))
    A, B, PW, ids = D["A_pre"].astype(np.float32), D["B_pre"].astype(np.float32), D["pwin"], D["ids"]
    on = PW >= np.percentile(PW, 80)
    ph = net.Photos("read")
    os.makedirs(OUTD, exist_ok=True)
    units = [("child", j, A[:, j]) for j in KIDS] + [("parent", 69, PW), ("coarse", 101, B[:, 101])]
    codes = np.random.default_rng(9).permutation(len(units))
    key = {}
    for (kind, n, X), cd in zip(units, codes):
        code = f"R{cd + 1:02d}"; key[code] = dict(kind=kind, neuron=int(n))
        r, c = ranked_pos(X); sheet(ph, ids, r, c, os.path.join(OUTD, f"{code}_pos.png"))
        if kind != "parent":
            r, c = ranked_neg_in(X, on); sheet(ph, ids, r, c, os.path.join(OUTD, f"{code}_negin.png"))
    json.dump(key, open(os.path.join(OUTD, "KEY.json"), "w"), indent=1)
    print(json.dumps(key))


if __name__ == "__main__":
    main()
