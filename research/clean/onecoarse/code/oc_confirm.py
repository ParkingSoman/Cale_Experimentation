"""Held-out confirmation packets (README rule 6). Photos: the 19,000 held + test photos (oc_split.npz), MINUS every
photo that appeared on any sheet or map I viewed (oc_split rows, oc_maps rows) and minus each member's top 60 (used
for the mean images). Band: ranks 60-250 of each pole after exclusion. 24 targets per pole on two sheets (rounds
c1, c2), distractors from the other members' same band where this member is below its median. Words: the final
words files given by OC_WORDS (positive, keys '<member>') and OC_NEGWORDS (negative). Controls: the same round's
words of another member.
Usage: OC_WORDS=judge/words_final.json OC_NEGWORDS=judge/words_neg_final.json python3 oc_confirm.py"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
CLEAN = os.path.join(ROOT, "..")
J = os.path.join(ROOT, "judge")
D = np.load(os.path.join(ROOT, "results", "oc_split.npz"))
M = np.load(os.path.join(ROOT, "results", "oc_maps.npz"))
IXT = np.load(os.path.join(CLEAN, "curves", "cache", "r18", "ix_train.npy"))
X = np.load(os.path.join(CLEAN, "hf", "imagenet", "val_224.npy"), mmap_mode="r")
MEM = ["p", "k305", "k97", "k23", "k297", "k163", "k275", "c101"]
S = 120
SEEN = set(D["rows"].tolist()) | set(M["rows"].tolist())
for n in MEM:
    SEEN |= set(np.argsort(-D[f"{n}_max"])[:60].tolist()) | set(np.argsort(D[f"{n}_min"])[:60].tolist())


def band(n, side):
    v = D[f"{n}_max"] if side == "pos" else -D[f"{n}_min"]
    o = [r for r in np.argsort(-v) if r not in SEEN]
    return np.array(o[:190]), D[f"{n}_amax"] if side == "pos" else D[f"{n}_amin"], v


def crop(r, cell, n):
    g, C = (14, 72) if n == "p" else (7, 112)
    raw = Image.fromarray(np.asarray(X[int(IXT[D["ix"][r]])]))
    cx, cy = (cell % g + .5) * 224 / g, (cell // g + .5) * 224 / g
    l, t = int(min(max(cx - C / 2, 0), 224 - C)), int(min(max(cy - C / 2, 0), 224 - C))
    return raw.crop((l, t, l + C, t + C)).resize((S, S))


def build(n, side, rnd, words, rng):
    own, am, v = band(n, side)
    tgt = rng.choice(own, 12, replace=False)
    pool = []
    for m in MEM:
        if m == n:
            continue
        o2, am2, _ = band(m, side)
        pool.append([(int(r), int(am2[r]), m) for r in o2 if v[r] < np.median(v) and r not in set(own.tolist())])
    dis, k = [], 0
    while len(dis) < 24 and any(pool):
        if pool[k % len(pool)]:
            dis.append(pool[k % len(pool)].pop(rng.integers(len(pool[k % len(pool)]))))
        k += 1
    items = [(int(r), int(am[r]), n, True) for r in tgt] + [(r, c, m, False) for r, c, m in dis]
    order = rng.permutation(len(items))
    img = Image.new("RGB", (6 * S, 6 * (S + 16)), "white")
    dr = ImageDraw.Draw(img)
    tg = []
    for num, i in enumerate(order, start=1):
        r, cell, who, is_t = items[i]
        x0, y0 = ((num - 1) % 6) * S, ((num - 1) // 6) * (S + 16)
        img.paste(crop(r, cell, who), (x0, y0 + 16))
        dr.text((x0 + 3, y0 + 2), str(num), fill="black")
        if is_t:
            tg.append(num)
    tag = rnd if side == "pos" else f"neg{rnd}"
    sheet = os.path.join(J, "packets", f"{n}_{tag}.png")
    img.save(sheet)
    ctrl = [m for m in MEM if m != n][rng.integers(len(MEM) - 1)]
    for item, w in ((f"{n}_{tag}_own", words[n]), (f"{n}_{tag}_ctrl", words[ctrl])):
        json.dump({"item": item, "kind": "pole", "words": w, "sheet": os.path.abspath(sheet)},
                  open(os.path.join(J, "packets", f"{item}.json"), "w"), indent=1)
    json.dump({"neuron": f"{n}_{tag}", "targets": tg, "control_words_from": ctrl, "photos": "held+test, unseen"},
              open(os.path.join(J, "truth", f"{n}_{tag}.json"), "w"), indent=1)


def main():
    pw = json.load(open(os.path.join(ROOT, os.environ["OC_WORDS"])))
    nw = json.load(open(os.path.join(ROOT, os.environ["OC_NEGWORDS"])))
    for rnd, seed in (("c1", 501), ("c2", 502)):
        rng = np.random.default_rng(seed)
        for n in MEM:
            build(n, "pos", rnd, pw, rng)
            build(n, "neg", rnd, nw, rng)
    print("excluded photos:", len(SEEN), "| packets:", 2 * 2 * 2 * len(MEM))


if __name__ == "__main__":
    main()
