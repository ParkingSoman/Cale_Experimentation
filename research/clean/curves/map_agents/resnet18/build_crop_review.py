# serves: H-map
"""A blind crop test, fixed before any scoring. It is the plan's replacement for the class test at stages 1 to 3.

For each card: 24 crops from SEALED photos on one sheet, in random order, labelled A to X. (The old cards were scored on 12 crops, with gaps in steps of 1/3.)
   12 fire    the node's own peak, on photos where its firing is at least half of its largest on the sealed photos.
   6 near     the peak of the node in the same layer most correlated with it over the sealed photos, on crops where this node's firing at the centre is under 20% of its largest.
   6 random   the same, for a random node in the same layer.
A reviewer sees only the words of a card and the sheet, and names the 12 that fire. Gap = share of the 12 firing crops named - share of the 12 others named.
Chance is 0. A channel passes at gap >= .2, which needs 8 of 12 (chance about .11 per channel). The line is 85% of channels.
Wrong cards: 1 item in 4 shows the words of another card from the same layer, on this node's sheet. The reviewer cannot tell which items these are. Their mean gap must be under .05.
Old cards read channel s3:17 as s3b1:17, as they were read then. Seed 0. The key stays in review2/<tag>/key.json, which a reviewer must not read.
Usage: build_crop_review.py TAG CARD_DIR [--stages 1,2,3]"""
import argparse
import glob
import json
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

from ev2 import CROP, grid, parse, value_field  # noqa: E402
from net import HERE, STORE, STRIDE, Photos, load  # noqa: E402

torch.set_num_threads(4)
M = load()

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("cards")
ap.add_argument("--stages", default="1,2,3,4")
a = ap.parse_args()
SEAL = os.path.join(STORE, "sealed")
OUT = os.path.join(HERE, "review2", a.tag)
os.makedirs(OUT, exist_ok=True)
random.seed(0)
rng = np.random.default_rng(0)
P = Photos("sealed")
stages_on = {int(t) for t in a.stages.split(",")}


def arr(n, kind):
    return np.asarray(np.load(os.path.join(SEAL, f"s{n['s']}b{n['b']}_{'i' if n['inner'] else 'h'}{kind}.npy"), mmap_mode="r"))


cards = {}
for f in sorted(glob.glob(os.path.join(a.cards, "*.jsonl"))):
    for line in open(f):
        if line.strip():
            d = json.loads(line)
            key = d.get("node") or d["channel"]
            if ":" in key and "b" not in key.split(":")[0] and not key.startswith("s0"):
                s, c = key.split(":")
                key = f"{s}b1:{c}"
            if key not in cards and parse(key)["s"] in stages_on:
                cards[key] = d
print(len(cards), "cards")


def crops(idx, pos, s):
    raw = P.raw(idx)
    g, half = grid(s), CROP[s] // 2
    out = []
    for j in range(len(idx)):
        r, c = int(pos[j]) // g, int(pos[j]) % g
        cy, cx = int((r + .5) * STRIDE[s]), int((c + .5) * STRIDE[s])
        r0, c0 = min(max(cy - half, 0), 224 - 2 * half), min(max(cx - half, 0), 224 - 2 * half)
        out.append(Image.fromarray(raw[j][r0:r0 + 2 * half, c0:c0 + 2 * half]).resize((160, 160)))
    return out


def layer_of(key):
    return key.split(":")[0]


items, key_out = [], {}
by_layer = {}
for k in cards:
    by_layer.setdefault(layer_of(k), []).append(k)
for num, (k, card) in enumerate(cards.items()):
    n = parse(k)
    hx, hp, hm = arr(n, "x"), arr(n, "p"), arr(n, "m")
    v = hx[:, n["c"]].astype(np.float64)
    big = v.max()
    fire = rng.permutation(np.where(v >= .5 * big)[0])[:12]
    if len(fire) < 12:
        fire = np.argsort(-v)[:12]
    hmf = hm.astype(np.float64)
    u = hmf[:, n["c"]] - hmf[:, n["c"]].mean()
    corr = (hmf - hmf.mean(0)).T @ u / (hmf.std(0) * u.std() * len(u) + 1e-9)
    corr[n["c"]] = -1

    def others(c, count, avoid):
        """Strong crops of channel c where THIS node is under 20% of its largest firing, at that crop's centre. Judged per crop: an early dial fires somewhere on every photo."""
        o = [int(i) for i in np.argsort(-hx[:, c].astype(np.float64))[:200] if int(i) not in avoid]
        picked = []
        for b in range(0, len(o), 32):
            sel = np.array(o[b:b + 32])
            with torch.no_grad():
                f = torch.relu(value_field(M, P.tensor(sel), n)).flatten(1).numpy()
            for j, i in enumerate(sel):
                if f[j, int(hp[i, c])] < .2 * big:
                    picked.append(int(i))
            if len(picked) >= count:
                break
        if len(picked) < count:
            return None
        return np.array(picked[:count], dtype=np.int64)

    # a channel too close to have crops where this node is quiet gives way to the next most correlated one
    near = None
    for near_c in np.argsort(-corr).tolist():
        if near_c != n["c"]:
            near = others(near_c, 6, set(fire.tolist()))
            if near is not None:
                break
    rnd = None
    for rand_c in rng.permutation(hx.shape[1]).tolist():
        if rand_c not in (n["c"], near_c):
            rnd = others(rand_c, 6, set(fire.tolist()) | set(near.tolist()))
            if rnd is not None:
                break
    tiles = crops(fire, hp[fire, n["c"]], n["s"]) + crops(near, hp[near, near_c], n["s"]) + crops(rnd, hp[rnd, rand_c], n["s"])
    truth = [True] * len(fire) + [False] * (len(near) + len(rnd))
    order = rng.permutation(len(tiles))
    sheet = Image.new("RGB", (6 * 164, 4 * 184), "white")
    d = ImageDraw.Draw(sheet)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWX"
    answer = []
    for pos_, t in enumerate(order):
        x, y = (pos_ % 6) * 164, (pos_ // 6) * 184
        d.text((x + 2, y + 2), letters[pos_], fill="black")
        sheet.paste(tiles[t], (x + 2, y + 18))
        if truth[t]:
            answer.append(letters[pos_])
    item = f"item{num:03d}"
    path = os.path.join(OUT, item + ".png")
    sheet.save(path)
    wrong = len(by_layer[layer_of(k)]) > 1 and random.random() < .25
    words_from = random.choice([q for q in by_layer[layer_of(k)] if q != k]) if wrong else k
    c = cards[words_from]
    items.append(dict(item=item, sheet=path, distinction=c.get("distinction", ""), positive=c.get("positive", ""), negative=c.get("negative", "")))
    key_out[item] = dict(node=k, stage=n["s"], layer=layer_of(k), fires=sorted(answer), wrong_card=wrong, words_from=words_from, near=near_c, random=rand_c)
order = list(range(len(items)))
random.shuffle(order)
json.dump([items[i] for i in order], open(os.path.join(OUT, "items.json"), "w"), indent=0)
json.dump(key_out, open(os.path.join(OUT, "key.json"), "w"), indent=0)
print(OUT, len(items), "items,", sum(v["wrong_card"] for v in key_out.values()), "wrong cards")
