"""S9: which automatic tie-break picks the judged parent among the 6 candidates (rule top-3 + |edge| top-3)?
Data: S6 (4) + S8 (19) judged first picks; S2 key (16) parents where present in the 6."""
import json, os, glob
import numpy as np
from convnext import *
M = np.load(os.path.join(HF, "imagenet", "m3_matrices.npz")); E = np.load(os.path.join(HF, "imagenet", "e1_edges.npz"))
mp = json.load(open(os.path.join(ROOT, "research/clean/map/network_map.json")))["neurons"]
H = [384]*3 + [768]*3 + [1536]*9 + [3072]*3; OFF = np.concatenate([[0], np.cumsum(H)])
truth = {}
k6 = json.load(open(os.path.join(RES, "vision-s6-key.json"))); p6 = json.load(open(os.path.join(ROOT, "research/clean/map/s6/picks.json")))
for k, p in p6.items(): truth[k] = tuple(k6[k][p[0]]["neuron"])
k8 = json.load(open(os.path.join(RES, "vision-s8-key.json"))); p8 = {}
for f in glob.glob(os.path.join(ROOT, "research/clean/map/s8/picks_*.json")): p8.update(json.load(open(f)))
for k, p in p8.items():
    if p: truth[k] = tuple(k8[k][p[0]]["neuron"])
KEY = {"1_351": (0, 245), "3_623": (2, 61), "4_499": (1, 365), "5_135": (4, 212), "8_4": (6, 205), "10_1316": (9, 686), "11_1031": (10, 1172),
       "12_136": (9, 1137), "14_1326": (13, 78), "14_33": (12, 548), "14_831": (3, 92), "15_16": (14, 1138), "16_384": (15, 1162), "16_26": (15, 2142),
       "16_2062": (13, 1136), "17_2283": (16, 2886)}
truth.update(KEY)
loc = lambda j: (int(np.searchsorted(OFF, j, side="right") - 1), int(j - OFF[np.searchsorted(OFF, j, side="right") - 1]))
res = {"in6": 0, "n": 0}; acc = {}
for k, t in truth.items():
    b, n = map(int, k.split("_")); e = E[str(b)][n]; rule = [int(OFF[p[0]] + p[1]) for p in mp[k]["parents"]]
    sc_rule = {int(OFF[p[0]] + p[1]): p[2] for p in mp[k]["parents"]}
    edge = [int(j) for j in np.argsort(-np.abs(e)) if j not in rule][:3]; cands = rule + edge
    tj = int(OFF[t[0]] + t[1]); res["n"] += 1
    if tj not in cands: continue
    res["in6"] += 1
    # features per candidate: rule score (C*log lift, recomputed for edge candidates), |edge| rank, block distance
    rate = M[f"{b}_rate"][n]; rs = {}
    for j in cands:
        best = 0
        for v in ("view", "point"):
            for s in ("pos", "neg"):
                C = float(M[f"{b}_{v}_{s}_C"][n, j]); br = float(M[f"{b}_{v}_{s}_br"][j]); L = C / max(br, 1e-6)
                if br > rate and L > 1: best = max(best, C * np.log(L))
        rs[j] = best
    er = {j: int((np.abs(e) > abs(e[j])).sum()) + 1 for j in cands}
    choosers = {
        "rule score": max(cands, key=lambda j: rs[j]),
        "|edge|": max(cands, key=lambda j: abs(e[j])),
        "rank sum": min(cands, key=lambda j: sorted(cands, key=lambda x: -rs[x]).index(j) + sorted(cands, key=lambda x: -abs(e[x])).index(j)),
        "rule score x log(1/edge rank)": max(cands, key=lambda j: rs[j] * np.log(1 + 1e4 / er[j])),
        "nearest block, then rule": max(cands, key=lambda j: (loc(j)[0], rs[j])),
    }
    for name, j in choosers.items(): acc[name] = acc.get(name, 0) + (j == tj)
print(f"judged parent inside the 6 candidates: {res['in6']} / {res['n']}")
for name, a in acc.items(): print(f"{name:32s}: picks the judged parent {a} / {res['in6']}")
json.dump({"in6": res, "choosers": acc}, open(os.path.join(RES, "vision-s9-choose.json"), "w"), indent=1)
