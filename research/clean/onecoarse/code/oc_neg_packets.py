"""Floor B3 (negative side) for the gate neurons of the core (n305, n163; README rule 5).
Reader sheet (held + test photos): the neuron's 20 strongest NEGATIVE cells by the input-only min (oc_split.npz bands:
top 10 and strong 10), crops centred on the min cell, no overlay, no positive side.
-> judge/negread/<n>.png (the reader agent sees only this).
Test sheet (train-cache photos, never shown to the reader): 12 of the neuron's 30 most negative cells (input-only min)
and 24 most-negative cells of the other core neurons and n101 where this neuron's min is above its median.
-> judge/packets/<n>_neg.png ; the key goes with the other keys. Words come later from the reader (judge/words_neg.json).
Usage: python3 oc_neg_packets.py sheets | packets"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "..")
J = os.path.join(HERE, "..", "judge")
R = os.path.join(HERE, "..", "results")
D = np.load(os.path.join(R, "oc_split.npz"))
F = np.load(os.path.join(R, "oc_floor.npz"))
IXT = np.load(os.path.join(CLEAN, "curves", "cache", "r18", "ix_train.npy"))
X = np.load(os.path.join(CLEAN, "hf", "imagenet", "val_224.npy"), mmap_mode="r")
GATES = ["k305", "k163"]
OTHERS = ["p", "k305", "k97", "k23", "k297", "k163", "k275", "c101"]
S, C = 120, 112


def crop(val_id, cell, g=7, C=C):
    raw = Image.fromarray(np.asarray(X[int(val_id)]))
    cx, cy = (cell % g + .5) * 224 / g, (cell // g + .5) * 224 / g
    l, t = int(min(max(cx - C / 2, 0), 224 - C)), int(min(max(cy - C / 2, 0), 224 - C))
    return raw.crop((l, t, l + C, t + C)).resize((S, S))


PT = None


def train_neg(n):
    global PT
    if n == "p":                                   # parent: input-only min on the train photos (oc_ptrain.npz), 14 x 14
        PT = PT if PT is not None else np.load(os.path.join(R, "oc_ptrain.npz"))
        assert (PT["train_ix"] == F["train_ix"]).all()
        return PT["zmin"].astype(np.float32), PT["znarg"].astype(int)
    if n == "c101":
        return F["cneg"].astype(np.float32), F["cnarg"].astype(int)
    j = int(n[1:])
    return F["tneg"][:, j].astype(np.float32), F["tnarg"][:, j].astype(int)


def sheets(names):
    os.makedirs(os.path.join(J, "negread"), exist_ok=True)
    for n in names:
        rows = D[f"{n}_bneg"][:20]
        g, cc = (14, 72) if n == "p" else (7, C)
        img = Image.new("RGB", (10 * S, 2 * S), "white")
        for k, r in enumerate(rows):
            img.paste(crop(IXT[D["ix"][r]], int(D[f"{n}_amin"][r]), g, cc), ((k % 10) * S, (k // 10) * S))
        img.save(os.path.join(J, "negread", f"{n}.png"))
        print(os.path.join(J, "negread", f"{n}.png"))


def packets():
    rnd = os.environ.get("OC_ROUND", "")                      # "" = first test; e.g. "r2" = re-test on new photos
    lo, hi = (int(x) for x in os.environ.get("OC_RANKS", "0,30").split(","))
    words = json.load(open(os.path.join(J, f"words_neg{('_' + rnd) if rnd else ''}.json")))
    rng = np.random.default_rng(31 + sum(map(ord, rnd)))
    keyd = os.path.join(J, "truth")
    names = list(words)
    only = os.environ.get("OC_ONLY", "").split(",") if os.environ.get("OC_ONLY") else names
    for i, n in enumerate(names):
        if n not in only:
            continue
        v, a = train_neg(n)
        own = np.argsort(v)[lo:hi]
        tgt = rng.choice(own, 12, replace=False)
        pool = []
        for m in OTHERS:
            if m == n:
                continue
            w, b = train_neg(m)
            cand = [r for r in np.argsort(w)[lo:hi] if v[r] > np.median(v) and r not in own]
            pool.append([(int(r), int(b[r]), m) for r in cand])
        dis, k = [], 0
        while len(dis) < 24 and any(pool):
            if pool[k % len(pool)]:
                dis.append(pool[k % len(pool)].pop(rng.integers(len(pool[k % len(pool)]))))
            k += 1
        items = [(int(r), int(a[r]), True, n) for r in tgt] + [(r, c, False, m) for r, c, m in dis]
        order = rng.permutation(len(items))
        img = Image.new("RGB", (6 * S, 6 * (S + 16)), "white")
        dr = ImageDraw.Draw(img)
        tg = []
        for num, ii in enumerate(order, start=1):
            r, cell, is_t, who = items[ii]
            x0, y0 = ((num - 1) % 6) * S, ((num - 1) // 6) * (S + 16)
            g, cc = (14, 72) if who == "p" else (7, C)
            img.paste(crop(IXT[F["train_ix"][r]], cell, g, cc), (x0, y0 + 16))
            dr.text((x0 + 3, y0 + 2), str(num), fill="black")
            if is_t:
                tg.append(num)
        tag = f"neg{rnd}"
        sheet = os.path.join(J, "packets", f"{n}_{tag}.png")
        img.save(sheet)
        allw = words if rnd else json.load(open(os.path.join(J, "words_neg.json")))   # controls from the same round
        ctrl = [m for m in allw if m != n][rng.integers(len(allw) - 1)]
        for item, w in ((f"{n}_{tag}_own", words[n]), (f"{n}_{tag}_ctrl", allw[ctrl])):
            json.dump({"item": item, "kind": "pole", "words": w, "sheet": os.path.abspath(sheet)},
                      open(os.path.join(J, "packets", f"{item}.json"), "w"), indent=1)
        json.dump({"neuron": f"{n}_{tag}", "targets": tg, "control_words_from": ctrl, "ranks": [lo, hi]},
                  open(os.path.join(keyd, f"{n}_{tag}.json"), "w"), indent=1)
    print("packets", 2 * len([n for n in names if n in only]))


if __name__ == "__main__":
    if sys.argv[1] == "sheets":
        sheets(sys.argv[2:] or GATES)
    else:
        packets()
