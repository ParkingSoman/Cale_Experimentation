"""Floor B packets (Mac). For each neuron with words in judge/words.json: a sheet of 36 numbered crops from the
train-cache photos (never shown to the reader): 12 of the neuron's own top-30 cells and 24 top-30 cells of its
siblings where the neuron itself is below its median. Packets: judge/packets/<item>.json + .png (own words), and a
control packet per neuron with a sibling's words (same sheet). Truth (which numbers are targets): judge/truth/.
Usage: python3 oc_packets.py"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "..")
J = os.path.join(HERE, "..", "judge")
F = np.load(os.path.join(HERE, "..", "results", "oc_floor.npz"))
IXT = np.load(os.path.join(CLEAN, "curves", "cache", "r18", "ix_train.npy"))
X = np.load(os.path.join(CLEAN, "hf", "imagenet", "val_224.npy"), mmap_mode="r")
WORDS = json.load(open(os.path.join(J, "words.json")))
ROUND = os.environ.get("OC_ROUND", "")                       # "" = round 1; "r2" = re-test of re-read neurons
LO, HI = (int(x) for x in os.environ.get("OC_RANKS", "0,30").split(","))   # rank window of the cells used
NEW = json.load(open(os.path.join(J, f"words_{ROUND}.json"))) if ROUND else {}
S = 120


def val(n):                                    # (per-photo max, its cell, grid, crop px)
    if n == "p":
        return F["ppos"].astype(np.float32), F["parg"].astype(int), 14, 72
    if n == "c101":
        return F["cpos"].astype(np.float32), F["carg"].astype(int), 7, 112
    j = int(n[1:])
    return F["tpos"][:, j].astype(np.float32), F["targ"][:, j].astype(int), 7, 112


def crop(r, cell, g, C):
    raw = Image.fromarray(np.asarray(X[int(IXT[F["train_ix"][r]])]))
    cx, cy = (cell % g + .5) * 224 / g, (cell // g + .5) * 224 / g
    l, t = int(min(max(cx - C / 2, 0), 224 - C)), int(min(max(cy - C / 2, 0), 224 - C))
    return raw.crop((l, t, l + C, t + C)).resize((S, S))


def build(n, rng):
    v, a, g, C = val(n)
    own = np.argsort(-v)[LO:HI]
    tgt = rng.choice(own, 12, replace=False)
    sibs = [m for m in WORDS if m != n]
    pool = []
    for m in sibs:
        w, b, gm, Cm = val(m)
        cand = [r for r in np.argsort(-w)[LO:HI] if v[r] < np.median(v) and r not in own]
        pool.append([(int(r), int(b[r]), gm, Cm) for r in cand])
    dis, k = [], 0
    while len(dis) < 24 and any(pool):
        if pool[k % len(pool)]:
            dis.append(pool[k % len(pool)].pop(rng.integers(len(pool[k % len(pool)]))))
        k += 1
    items = [(int(r), int(a[r]), g, C, True) for r in tgt] + [(*d, False) for d in dis]
    order = rng.permutation(len(items))
    img = Image.new("RGB", (6 * S, 6 * (S + 16)), "white")
    dr = ImageDraw.Draw(img)
    truth = []
    for num, i in enumerate(order, start=1):
        r, cell, gg, CC, is_t = items[i]
        x0, y0 = ((num - 1) % 6) * S, ((num - 1) // 6) * (S + 16)
        img.paste(crop(r, cell, gg, CC), (x0, y0 + 16))
        dr.text((x0 + 3, y0 + 2), str(num), fill="black")
        if is_t:
            truth.append(num)
    return img, truth


def main():
    rng = np.random.default_rng(22 if not ROUND else 20 + sum(map(ord, ROUND)))
    for d in ("packets", "truth", "answers"):
        os.makedirs(os.path.join(J, d), exist_ok=True)
    names = list(WORDS)
    build_for = [k.rsplit("_", 1)[0] for k in NEW] if ROUND else names
    if os.environ.get("OC_ONLY"):
        build_for = [n for n in build_for if n in os.environ["OC_ONLY"].split(",")]
    tag = f"_{ROUND}" if ROUND else ""
    for i, n in enumerate(names):
        if n not in build_for:
            continue
        img, truth = build(n, rng)
        sheet = os.path.join(J, "packets", f"{n}{tag}.png")
        img.save(sheet)
        ctrl = names[(i + 1 + rng.integers(len(names) - 1)) % len(names)]
        ctrl = ctrl if ctrl != n else names[(i + 1) % len(names)]
        own_words = NEW[n + tag] if ROUND else WORDS[n]
        ctrl_words = NEW.get(ctrl + tag, WORDS[ctrl]) if ROUND else WORDS[ctrl]   # noun-free rounds use noun-free controls
        for item, words in ((f"{n}{tag}_own", own_words), (f"{n}{tag}_ctrl", ctrl_words)):
            json.dump({"item": item, "kind": "pole", "words": words, "sheet": os.path.abspath(sheet)},
                      open(os.path.join(J, "packets", f"{item}.json"), "w"), indent=1)
        json.dump({"neuron": n + tag, "targets": truth, "control_words_from": ctrl, "ranks": [LO, HI]},
                  open(os.path.join(J, "truth", f"{n}{tag}.json"), "w"), indent=1)
    print("packets", 2 * len(build_for))


if __name__ == "__main__":
    main()
