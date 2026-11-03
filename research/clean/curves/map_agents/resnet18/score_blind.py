# serves: H-map
"""Score the blind reviewers on SEALED photos. Lines fixed before the picks were made (plan Step 4).

Test A, specific enough to lose. For each channel: percentile of its largest firing at block 1 over the sealed photos, mean over a
class's photos. Gap = mean over HIGH classes - mean over NEAR classes. A pick whose class name appears in the channel's own card text
is dropped, since the explanation must predict beyond what it names. A channel with fewer than 2 picks left on a side is not scored.
PASS for the map: at least 85% of channels have gap >= .2. Reported per stage. Chance: gap 0.
Test B, real parents. Share of links where the reviewer chose the true parent of 5. PASS: at least 70%. Chance: 20%.
Reported apart for links where one agent wrote both cards, since shared wording could give the answer away.
Usage: score_blind.py a|b"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from net import HERE, STORE, class_names, parse  # noqa: E402

REV = os.path.join(HERE, "review")
SEAL = os.path.join(STORE, "sealed")
key = json.load(open(os.path.join(REV, "key.json")))
which = sys.argv[1]
if which == "a":
    pack = {d["id"]: d for d in json.load(open(os.path.join(REV, "pack_a.json")))}
    names = class_names()
    lab = np.load(os.path.join(SEAL, "labels.npy"))
    rows = []
    for line in open(os.path.join(REV, "picks_a.jsonl")):
        if not line.strip():
            continue
        d = json.loads(line)
        ch = key["a"][d["id"]]
        s, c = parse(ch)
        v = np.asarray(np.load(os.path.join(SEAL, f"s{s}b1_hx.npy"), mmap_mode="r")[:, c], dtype=np.float64)
        rank = v.argsort().argsort() / (len(v) - 1)
        txt = " ".join(pack[d["id"]][k] for k in ("distinction", "positive", "negative")).lower()

        def clean(picks):
            out = []
            for q in picks:
                nm = names[int(q)].lower()
                if nm in txt or any(w in txt for w in nm.split() if len(w) > 4):
                    continue
                out.append(int(q))
            return out

        hi, ne = clean(d["high"]), clean(d["near"])
        if len(hi) < 2 or len(ne) < 2:
            rows.append(dict(channel=ch, gap=None, high=None, near=None, kept=(len(hi), len(ne))))
            continue
        h = float(np.mean([rank[lab == q].mean() for q in hi]))
        n = float(np.mean([rank[lab == q].mean() for q in ne]))
        rows.append(dict(channel=ch, gap=h - n, high=h, near=n, kept=(len(hi), len(ne))))
    json.dump(rows, open(os.path.join(HERE, "results", "blind_a.json"), "w"), indent=0)
    done = [r for r in rows if r["gap"] is not None]
    print(f"picked {len(rows)} of {len(pack)}, scored {len(done)}; pass share at gap >= .2: {np.mean([r['gap'] >= .2 for r in done]):.3f}; line is .85")
    print(f"mean high {np.mean([r['high'] for r in done]):.3f}, mean near {np.mean([r['near'] for r in done]):.3f}")
    for st in ("s4", "s3", "s2", "s1"):
        g = [r for r in done if r["channel"].startswith(st)]
        if g:
            print(f"{st}: n {len(g)}, pass {np.mean([r['gap'] >= .2 for r in g]):.3f}, at gap >= .1 {np.mean([r['gap'] >= .1 for r in g]):.3f}, mean gap {np.mean([r['gap'] for r in g]):.3f}")
    print("worst:", sorted([(round(r["gap"], 2), r["channel"]) for r in done])[:10])
else:
    rows = []
    for line in open(os.path.join(REV, "picks_b.jsonl")):
        if not line.strip():
            continue
        d = json.loads(line)
        k = key["b"][d["id"]]
        rows.append(dict(id=d["id"], child=k["child"], parent=k["parent"], right=d["pick"] == k["answer"], same_agent=k["same_agent"]))
    json.dump(rows, open(os.path.join(HERE, "results", "blind_b.json"), "w"), indent=0)
    print(f"links {len(rows)}: right {np.mean([r['right'] for r in rows]):.3f}; line is .70, chance .20")
    for name, g in (("same agent wrote both", [r for r in rows if r["same_agent"]]), ("different agents", [r for r in rows if not r["same_agent"]])):
        if g:
            print(f"  {name}: n {len(g)}, right {np.mean([r['right'] for r in g]):.3f}")
    for st in ("s4", "s3", "s2"):
        g = [r for r in rows if r["child"].startswith(st)]
        if g:
            print(f"  child in {st}: n {len(g)}, right {np.mean([r['right'] for r in g]):.3f}")
