# serves: H-map
"""Score the round 2 cards on SEALED photos, one convolution at a time. Lines and predictions: results/PREDICTIONS.md, written before any card existed.
At each node's own peak on its 64 strongest sealed photos, with the exact inputs of Definition DC4:
   Tree cover     the main parent's root mean square change is at least 2 times the mean of 3 matched controls. Line: 90% of nodes. No main parent, or one that is not a direct input, is a miss.
   Parents act    the same test on each named input. Line: 90% of links.
   Parents carry  explained = 1 - E (sum of the inputs outside the named list)^2 / E (sum of all inputs)^2. Pass at >= .25 and >= 3 times a random list of the same size. Line: 80% of children.
Matched controls: direct inputs with the closest spread of largest firing over the sealed photos, outside the node's top 30 inputs on these photos.
--pack builds the blind pack for "Real parents": a child's words and 5 candidates' words, one true main parent, 4 others of the parent's layer that this card does not name. Only children whose main parent has a card. Seed 0.
Usage: score_cards2.py [--pack] [--dir cards2]"""
import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ev2 import exact_inputs, parse  # noqa: E402
from net import HERE, STORE, Photos, load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pack", action="store_true")
ap.add_argument("--dir", default="cards2")
a = ap.parse_args()
cards = {}
for f in sorted(glob.glob(os.path.join(HERE, a.dir, "*.jsonl"))):
    for line in open(f):
        if line.strip():
            d = json.loads(line)
            d["agent"] = os.path.basename(f)[:-6]
            cards.setdefault(d["node"], d)
print(len(cards), "cards")


def clean(t):
    return (t or "").replace(" (skip)", "").strip()


if a.pack:
    random.seed(0)
    out = os.path.join(HERE, "review2", "parents")
    os.makedirs(out, exist_ok=True)
    by_layer = {}
    for k in cards:
        by_layer.setdefault(k.split(":")[0], []).append(k)
    items, key = [], {}

    def words(c):
        return " ".join(str(c.get(f, "")) for f in ("distinction", "positive", "negative"))

    for k, c in cards.items():
        mp = c.get("main_parent") or {}
        p = clean(mp.get("node"))
        if p not in cards:
            continue
        named = {clean(i.get("node")) for i in c.get("inputs", [])} | {p}
        pool = [q for q in by_layer[p.split(":")[0]] if q not in named and q != k]
        if len(pool) < 4:
            continue
        cand = random.sample(pool, 4) + [p]
        random.shuffle(cand)
        item = f"parent{len(items):03d}"
        items.append(dict(item=item, child=words(c), candidates={"ABCDE"[i]: words(cards[q]) for i, q in enumerate(cand)}))
        key[item] = dict(child=k, parent=p, answer="ABCDE"[cand.index(p)], same_agent=cards[p]["agent"] == c["agent"])
    json.dump(items, open(os.path.join(out, "items.json"), "w"), indent=0)
    json.dump(key, open(os.path.join(out, "key.json"), "w"), indent=0)
    print(len(items), "items in", out)
    sys.exit()

torch.set_num_threads(4)
m = load()
P = Photos("sealed")
SEAL = os.path.join(STORE, "sealed")
rng = np.random.default_rng(0)


def col(n, kind):
    key = f"s{n['s']}b{n['b']}_{'i' if n['inner'] else 'h'}{kind}"
    return np.load(os.path.join(SEAL, key + ".npy"), mmap_mode="r")[:, n["c"]]


rows, links = [], []
for k, c in cards.items():
    n = parse(k)
    v = np.asarray(col(n, "x"), dtype=np.float64)
    idx = np.sort(np.argsort(-v)[:64])
    pos = np.asarray(col(n, "p")[idx]).astype(np.int64)
    D, names, _, _ = exact_inputs(m, P.tensor(idx), n, pos)
    tags = [clean(t) for t in names]
    rms = np.sqrt((D ** 2).mean(0))
    wt = np.abs(D).mean(0)
    spread = np.array([float(np.std(col(parse(t), "x"))) for t in tags])
    barred = set(np.argsort(-wt)[:30].tolist())

    def acts(name):
        if name not in tags:
            return None
        j = tags.index(name)
        ctrl = [int(q) for q in np.argsort(np.abs(spread - spread[j])) if int(q) not in barred and int(q) != j][:3]
        return float(rms[j]), float(rms[ctrl].mean())

    mp = clean((c.get("main_parent") or {}).get("node"))
    res = acts(mp) if mp else None
    tree = res is not None and res[0] >= 2 * res[1]
    named = [mp] if mp else []
    for i in c.get("inputs", []):
        t = clean(i.get("node"))
        if t and t not in named:
            named.append(t)
    in_tags = [t for t in named if t in tags]
    for t in named:
        r = acts(t)
        links.append(dict(child=k, parent=t, direct=r is not None, ok=bool(r and r[0] >= 2 * r[1]), rms=r[0] if r else None, control=r[1] if r else None))
    sig = float((D.sum(1) ** 2).mean()) + 1e-12

    def explained(js):
        mask = np.ones(D.shape[1], bool)
        mask[js] = False
        return 1 - float((D[:, mask].sum(1) ** 2).mean()) / sig

    ex = explained([tags.index(t) for t in in_tags]) if in_tags else 0.0
    rand = float(np.mean([explained(rng.choice(D.shape[1], len(in_tags), replace=False)) for _ in range(20)])) if in_tags else 0.0
    carry = ex >= .25 and ex >= 3 * max(rand, 1e-9)
    # added after the first scoring: "Parents act" passed 371 of 371, so it does not discriminate. These are descriptive, not lines.
    order = np.argsort(-wt)
    code = explained(order[:len(in_tags)].tolist()) if in_tags else 0.0
    rank = int(np.where(order == tags.index(mp))[0][0]) + 1 if mp in tags else None
    rows.append(dict(node=k, layer=k.split(":")[0], agent=c["agent"], main=mp, tree=bool(tree), named=len(named), direct=len(in_tags), explained=ex, random=rand, carry=bool(carry), sure=c.get("sure"), code_list=code, main_rank=rank))
    print(k, "main", mp, "tree", tree, f"named {len(named)} direct {len(in_tags)} explained {ex:.2f} random {rand:.2f}", flush=True)

for layer in sorted({r["layer"] for r in rows}) + ["all"]:
    rr = [r for r in rows if layer == "all" or r["layer"] == layer]
    ll = [x for x in links if layer == "all" or x["child"].split(":")[0] == layer]
    print(f"{layer}: {len(rr)} nodes. Tree cover {np.mean([r['tree'] for r in rr]):.2f} (line .90). Parents act {np.mean([x['ok'] for x in ll]):.2f} of {len(ll)} links (line .90). "
          f"Parents carry {np.mean([r['carry'] for r in rr]):.2f} (line .80). Median explained {np.median([r['explained'] for r in rr]):.2f}. Named links that are not direct inputs: {sum(not x['direct'] for x in ll)}")
st4 = [r for r in rows if r["layer"].startswith("s4")]
ranks = [r["main_rank"] for r in st4 if r["main_rank"]]
print(f"stage 4 only: {len(st4)} nodes. Tree cover {np.mean([r['tree'] for r in st4]):.3f}. Parents carry {np.mean([r['carry'] for r in st4]):.3f}. No main parent or not direct: {sum(r['main_rank'] is None for r in st4)}")
print(f"main parent's rank by effect: median {np.median(ranks):.0f}, rank 1 on {np.mean([q == 1 for q in ranks]):.2f}, top 5 on {np.mean([q <= 5 for q in ranks]):.2f}. Named list explains {np.median([r['explained'] for r in st4]):.2f}, code's top list of the same size {np.median([r['code_list'] for r in st4]):.2f}")
json.dump(dict(nodes=rows, links=links), open(os.path.join(HERE, "results", "cards2_scores.json"), "w"), indent=0)
