# serves: H-map
"""Build the blind review packs from the cards, before any scoring. Seed 0. Opaque ids hide channel numbers.

Pack A, specific enough to lose: each channel's distinction, positive and negative text, nothing else.
Pack B, real parents, 1 of 5: a child's text and 5 candidates: its true parent and 4 distractors of the parent's stage. Only links
whose parent has a card. Distractors are never a listed parent of this child. Up to 2 come from the same agent as the true parent,
since one agent's wording could give the answer away.
A channel read by two agents uses its first card. The key stays in review/key.json, which a reviewer must not read."""
import glob
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.join(HERE, "review")
random.seed(0)
cards = {}
for f in sorted(glob.glob(os.path.join(HERE, "cards", "*.jsonl"))):
    agent = os.path.basename(f)[:-6]
    for line in open(f):
        if line.strip():
            d = json.loads(line)
            d["agent"] = agent
            d.setdefault("all_parents", [])
            if d["channel"] in cards:
                cards[d["channel"]]["parents"] += d.get("parents", [])
            else:
                cards[d["channel"]] = d


def text(d):
    return {"distinction": d["distinction"], "positive": d["positive"], "negative": d["negative"]}


chans = sorted(cards)
order = chans[:]
random.shuffle(order)
ids = {c: f"c{i:03d}" for i, c in enumerate(order)}
pack_a = [{"id": ids[c], **text(cards[c])} for c in order]
key = {"a": {ids[c]: c for c in order}, "b": {}}
pack_b = []
for c in order:
    listed = {p["channel"] for p in cards[c].get("parents", [])}
    for p in cards[c].get("parents", []):
        q = p["channel"]
        if q not in cards:
            continue
        stage = q.split(":")[0]
        pool = [x for x in chans if x.split(":")[0] == stage and x not in listed and x != c]
        same = [x for x in pool if cards[x]["agent"] == cards[q]["agent"]]
        other = [x for x in pool if cards[x]["agent"] != cards[q]["agent"]]
        random.shuffle(same)
        random.shuffle(other)
        dist = same[:2]
        dist += other[:4 - len(dist)]
        if len(dist) < 4:
            continue
        cand = dist + [q]
        random.shuffle(cand)
        tid = f"t{len(pack_b):03d}"
        pack_b.append({"id": tid, "child": text(cards[c]), "candidates": {"ABCDE"[i]: text(cards[x]) for i, x in enumerate(cand)}})
        key["b"][tid] = {"child": c, "parent": q, "answer": "ABCDE"[cand.index(q)], "same_agent": cards[c]["agent"] == cards[q]["agent"]}
os.makedirs(REV, exist_ok=True)
json.dump(pack_a, open(os.path.join(REV, "pack_a.json"), "w"), indent=0)
json.dump(pack_b, open(os.path.join(REV, "pack_b.json"), "w"), indent=0)
json.dump(key, open(os.path.join(REV, "key.json"), "w"), indent=0)
links = [(c, p["channel"], p.get("sign", "+")) for c in chans for p in cards[c].get("parents", [])]
json.dump(sorted(set(links)), open(os.path.join(HERE, "results", "links.json"), "w"), indent=0)
print(f"pack A {len(pack_a)} channels; pack B {len(pack_b)} links; all links {len(set(links))}")
