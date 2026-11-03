# serves: H-map
"""Line "Families act" of Step 4, on SEALED photos. Fixed before the picks were scored.

For each family: its channels go to their means through their stages (Definition DC1). KL against N per class, over 3 sealed photos per class.
Controls: 2 sets of the same number of random channels per stage, not in the family.
A reviewer named 10 classes from the words alone, before this ran (review/picks_c.jsonl).
PASS for a family: at least 6 of the 10 are among the top 100 losing classes of 1000 (chance is 1), and the same 10 classes lose at least
2 times more under the family than under the controls.
PASS for the line: most families pass.
PREDICTION, written before the run: tight families pass, broad ones do not. About half pass in all. The tone family (9 channels) and the
strand family pass. The two families with the lowest "sure", "furry face" and "stroke pairs", do not.
Usage: score_family.py [--smoke]"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from net import HERE, STORE, WIDTH, Photos, block_means, forward, load, parse  # noqa: E402

smoke = "--smoke" in sys.argv
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MEANS = block_means(os.path.join(STORE, "read"))
P = Photos("sealed")
idx = P.per_class(3)
if smoke:
    idx = idx[::40]
lab = P.lab[idx]
REV = os.path.join(HERE, "review")
key = json.load(open(os.path.join(REV, "key_c.json")))
picks = {json.loads(l)["id"]: json.loads(l)["classes"] for l in open(os.path.join(REV, "picks_c.jsonl")) if l.strip()}
m = load().to(DEV)
base = []
with torch.no_grad():
    for b in range(0, len(idx), 64):
        y, _ = forward(m, P.tensor(idx[b:b + 64]).to(DEV))
        base.append(F.log_softmax(y, 1).float().cpu())
base = torch.cat(base)


def per_class_kl(chs):
    fill = {}
    for s, c in chs:
        fill.setdefault(s, []).append(c)
    f = {s: (torch.tensor(cs, dtype=torch.long, device=DEV), MEANS[s][0][cs].to(DEV), MEANS[s][1][cs].to(DEV)) for s, cs in fill.items()}
    out = []
    with torch.no_grad():
        for b in range(0, len(idx), 64):
            y, _ = forward(m, P.tensor(idx[b:b + 64]).to(DEV), f)
            out.append(F.log_softmax(y, 1).float().cpu())
    q = torch.cat(out)
    kl = (base.exp() * (base - q)).sum(1).numpy()
    return np.array([kl[lab == c].mean() if (lab == c).any() else 0.0 for c in range(1000)]), float(kl.mean())


rows = []
for fid, k in key.items():
    chs = [parse(c) for c in k["channels"]]
    byc, total = per_class_kl(chs)
    top100 = set(np.argsort(-byc)[:100].tolist())
    pick = picks[fid]
    hits = sum(int(c) in top100 for c in pick)
    ctrl_on_pick, ctrl_total = [], []
    for seed in range(2):
        rng = np.random.default_rng(seed)
        cs = []
        for s, c in chs:
            while True:
                q = int(rng.integers(WIDTH[s]))
                if (s, q) not in chs and (s, q) not in cs:
                    cs.append((s, q))
                    break
        b2, t2 = per_class_kl(cs)
        ctrl_on_pick.append(float(b2[pick].mean()))
        ctrl_total.append(t2)
    on_pick = float(byc[pick].mean())
    ok = hits >= 6 and on_pick >= 2 * float(np.mean(ctrl_on_pick))
    rows.append(dict(id=fid, name=k["name"], n=len(chs), hits=hits, on_pick=on_pick, ctrl_on_pick=float(np.mean(ctrl_on_pick)), total=total, ctrl_total=float(np.mean(ctrl_total)), ok=bool(ok)))
    print(f"{k['name'][:44]:44s} n {len(chs)} hits {hits}/10 KL on picks {on_pick:.3f} controls {np.mean(ctrl_on_pick):.3f} whole-list KL {total:.3f} controls {np.mean(ctrl_total):.3f} {'PASS' if ok else 'fail'}", flush=True)
json.dump(rows, open(os.path.join(HERE, "results", "family_sealed" + ("_smoke" if smoke else "") + ".json"), "w"), indent=0)
print(f"families {len(rows)}: pass {np.mean([r['ok'] for r in rows]):.2f}; mean hits {np.mean([r['hits'] for r in rows]):.1f} of 10 (chance 1); KL family / control, median {np.median([r['total'] / max(r['ctrl_total'], 1e-9) for r in rows]):.2f}")
