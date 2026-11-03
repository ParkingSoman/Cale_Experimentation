# serves: H-map
"""Score the blind crop test of build_crop_review.py. Usage: score_crop_review.py TAG. Reads review2/TAG/picks_*.jsonl: {"item": ..., "fires": ["A", ...]}."""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "review2", sys.argv[1])
key = json.load(open(os.path.join(OUT, "key.json")))
picks = {}
for f in glob.glob(os.path.join(OUT, "picks_*.jsonl")):
    for line in open(f):
        if line.strip():
            d = json.loads(line)
            picks[d["item"]] = d["fires"]
rows = []
for item, k in key.items():
    if item not in picks:
        continue
    named, true = set(picks[item]), set(k["fires"])
    others = k.get("crops", 24) - len(true)
    rows.append(dict(item=item, node=k["node"], stage=k["stage"], layer=k["layer"], wrong=k["wrong_card"],
                     gap=len(named & true) / len(true) - len(named - true) / others))
print(f"{len(rows)} of {len(key)} items answered")
real = [r for r in rows if not r["wrong"]]
wrong = [r for r in rows if r["wrong"]]
for s in sorted({r["stage"] for r in rows}):
    v = [r["gap"] for r in real if r["stage"] == s]
    w = [r["gap"] for r in wrong if r["stage"] == s]
    print(f"stage {s}: {len(v)} cards, pass {np.mean([g >= .2 for g in v]):.2f}, mean gap {np.mean(v):.2f}; wrong cards {len(w)}, mean gap {np.mean(w) if w else float('nan'):.2f}")
print(f"all: {len(real)} cards, pass {np.mean([r['gap'] >= .2 for r in real]):.3f} (line .85), mean gap {np.mean([r['gap'] for r in real]):.3f}")
if wrong:
    print(f"wrong cards: {len(wrong)}, mean gap {np.mean([r['gap'] for r in wrong]):.3f} (line under .05)")
json.dump(rows, open(os.path.join(OUT, "scores.json"), "w"), indent=0)
