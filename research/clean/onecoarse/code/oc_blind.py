"""Copy one round's packets to judge/blind/ under neutral codes, add them to the code map in judge/truth/codes.json,
and split them between two judges so that no judge sees the same sheet twice.
Usage: python3 oc_blind.py r2   (prints each judge's packet list)"""
import json
import os
import random
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
J = os.path.join(HERE, "..", "judge")
rnd = sys.argv[1]
random.seed(hash(rnd) % 1000)
items = sorted(f[:-5] for f in os.listdir(os.path.join(J, "packets")) if f.endswith(f"_{rnd}_own.json") or f.endswith(f"_{rnd}_ctrl.json"))
cmap_path = os.path.join(J, "truth", "codes.json")
cmap = json.load(open(cmap_path))
used = set(cmap)
codes = [f"{rnd}q{i:02d}" for i in range(1, len(items) + 1)]
random.shuffle(codes)
sheets = sorted(set(json.load(open(os.path.join(J, "packets", f"{i}.json")))["sheet"] for i in items))
scode = {s: f"{rnd}s{k:02d}.png" for k, s in enumerate(random.sample(sheets, len(sheets)), 1)}
by_sheet = {}
for it, c in zip(items, codes):
    assert c not in used
    p = json.load(open(os.path.join(J, "packets", f"{it}.json")))
    sh = os.path.abspath(os.path.join(J, "blind", scode[p["sheet"]]))
    if not os.path.exists(sh):
        shutil.copy(p["sheet"], sh)
    json.dump({"item": c, "kind": "pole", "words": p["words"], "sheet": sh}, open(os.path.join(J, "blind", f"{c}.json"), "w"), indent=1)
    cmap[c] = it
    by_sheet.setdefault(it.rsplit("_", 1)[0], []).append(c)
json.dump(cmap, open(cmap_path, "w"), indent=1)
j1, j2 = [], []
for n, cs in sorted(by_sheet.items()):
    random.shuffle(cs)
    j1.append(cs[0]); j2.append(cs[1])
print("judge1", " ".join(os.path.abspath(os.path.join(J, "blind", f"{c}.json")) for c in sorted(j1)))
print("judge2", " ".join(os.path.abspath(os.path.join(J, "blind", f"{c}.json")) for c in sorted(j2)))
