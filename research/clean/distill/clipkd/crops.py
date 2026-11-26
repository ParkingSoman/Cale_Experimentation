"""CALC 3.6 crop sheets for the neuron checklist (steps 1-4), from out/poles.npz. FIT photos only (positions 0-3).

Per unit and pole, one sheet:
  row 0: intersection images of the top 40 crops: aligned colour mean, edge mean (checklist step 2)
  rows : TOP band (ranks 0-11), STRONG band (8 from ranks 12-79), WEAK band (6 from ranks 400-799)
  tile = 64 x 64 crop centred on the unit's token (shown at 128) above the whole photo with the patch boxed.
  A class-token crop (token 0) shows the whole photo with a full-frame box.
Units get anonymous codes; the code -> unit key is in cards/KEY.json (readers never open it).
Readers: reader r reads the pole of one unit per group, roles rotated, so no reader sees a student and its own members.
Usage: PYTHONPATH=distill/clipkd/pylib curves/.venv/bin/python -I distill/clipkd/crops.py
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_clip import OUT, net  # noqa: E402

CARDS = os.path.join(HERE, "cards")
T = 128


def window(img, tok, half=32):
    if tok == 0:
        return img
    r, c = divmod(int(tok) - 1, 14)
    cy, cx = int((r + .5) * 16) + half, int((c + .5) * 16) + half
    p = np.pad(img, ((half, half), (half, half), (0, 0)), mode="reflect")
    return p[cy - half:cy + half, cx - half:cx + half]


def tile(img, tok):
    t = Image.new("RGB", (T, T + T // 2 + 4), "white")
    t.paste(Image.fromarray(window(img, tok)).resize((T, T)), (0, 0))
    ctx = Image.fromarray(img).resize((T // 2, T // 2))
    d = ImageDraw.Draw(ctx)
    if tok == 0:
        d.rectangle([0, 0, T // 2 - 1, T // 2 - 1], outline="red", width=2)
    else:
        r, c = divmod(int(tok) - 1, 14)
        s = T / 2 / 224
        d.rectangle([c * 16 * s - 4, r * 16 * s - 4, (c + 1) * 16 * s + 4, (r + 1) * 16 * s + 4], outline="red", width=1)
    t.paste(ctx, (0, T + 4))
    return t


def means(imgs, toks):
    W = [window(i, t).astype(np.float32) for i, t in zip(imgs, toks) if t != 0]
    if len(W) < 5:  # class-token unit: mean of whole photos, shrunk to the crop size
        W = [np.asarray(Image.fromarray(i).resize((64, 64)), np.float32) for i in imgs]
    W = np.stack(W)
    col = W.mean(0)
    g = W.mean(3)
    gy, gx = np.gradient(g, axis=(1, 2))
    e = np.hypot(gx, gy).mean(0)
    e = 255 * e / (e.max() + 1e-6)
    return (Image.fromarray(col.clip(0, 255).astype(np.uint8)).resize((T, T)),
            Image.fromarray(e.astype(np.uint8)).resize((T, T)).convert("RGB"))


def ranked(P, k, pole, rows):
    v = P["gmax"][rows, k] if pole == 0 else -P["pmin"][rows, k]
    tok = P["gtok"][rows, k] if pole == 0 else P["ptok"][rows, k]
    o = np.argsort(-v)
    return rows[o], tok[o]


def sheet(P, ph, k, pole, rows, path):
    r, t = ranked(P, k, pole, rows)
    ids = P["photo_ids"]
    pick = list(range(12)) + list(np.linspace(12, 79, 8).astype(int)) + list(np.linspace(400, 799, 6).astype(int))
    imgs = ph.raw(ids[r[:40]])
    m_col, m_edge = means(imgs, t[:40])
    th = T + T // 2 + 4
    lines = [("MEAN of top 40 crops: colour, edges", None), ("TOP band, ranks 1-6", pick[0:6]), ("TOP band, ranks 7-12", pick[6:12]),
             ("STRONG band (ranks 13-80)", pick[12:20]), ("WEAK band (ranks 401-800)", pick[20:26])]
    im = Image.new("RGB", (8 * (T + 4), sum(T + 20 if s is None else th + 20 for _, s in lines)), "white")
    d = ImageDraw.Draw(im)
    y = 0
    for lab, sel in lines:
        d.text((4, y + 4), lab, fill="black")
        y += 18
        if sel is None:
            im.paste(m_col, (0, y)); im.paste(m_edge, (T + 4, y)); y += T + 2
            continue
        raws = ph.raw(ids[r[sel]])
        for c, (img, j) in enumerate(zip(raws, sel)):
            im.paste(tile(img, t[j]), (c * (T + 4), y))
        y += th + 2
    im.save(path)


def main():
    P = dict(np.load(os.path.join(OUT, "poles.npz")))
    ph = net.Photos("read")
    fit = np.where(P["pos"] < 4)[0]
    os.makedirs(CARDS, exist_ok=True)
    named = np.where(P["unit_role"] != "random")[0]
    rng = np.random.default_rng(5)
    codes = rng.permutation(len(named))
    key, plan = {}, {"pos": {}, "neg": {}}
    roles = ["student", "single", "m2", "m3"]
    for k, cd in zip(named, codes):
        code = f"U{cd + 1:02d}"
        b = int(P["unit_block"][k]) + 1
        key[code] = dict(unit=int(k), model=str(P["unit_model"][k]), block=b, neuron=int(P["unit_neuron"][k]),
                         role=str(P["unit_role"][k]), group=int(P["unit_group"][k]))
        for pole, nm in ((0, "pos"), (1, "neg")):
            sheet(P, ph, k, pole, fit, os.path.join(CARDS, f"{code}_{nm}.png"))
            g, ri = key[code]["group"], roles.index(key[code]["role"])
            reader = (ri + g + (0 if pole == 0 else 2)) % 4
            plan[nm].setdefault(f"reader{reader}", []).append(code)
    json.dump(key, open(os.path.join(CARDS, "KEY.json"), "w"), indent=1)
    json.dump(plan, open(os.path.join(CARDS, "READERS.json"), "w"), indent=1)
    print(json.dumps(plan))


if __name__ == "__main__":
    main()
