"""Map v2: parent of every neuron (blocks 1-17) chosen among rule top-3 (C*log lift) + |edge| top-3 by
rule score x log(1 + 1e4 / edge rank). Validated: pick inside the blind judge's top-3 for 22/23 (S9; chosen post hoc
among 3 choosers). Output research/clean/map/network_map_v2.json + tree stats."""
import json, os
import numpy as np
from convnext import *
M = np.load(os.path.join(HF, "imagenet", "m3_matrices.npz")); E = np.load(os.path.join(HF, "imagenet", "e1_edges.npz"))
mp = json.load(open(os.path.join(ROOT, "research/clean/map/network_map.json")))["neurons"]
H = [384]*3 + [768]*3 + [1536]*9 + [3072]*3; OFF = np.concatenate([[0], np.cumsum(H)]); VS = [(v, s) for v in ("view", "point") for s in ("pos", "neg")]
loc = lambda j: (int(np.searchsorted(OFF, j, side="right") - 1), int(j - OFF[np.searchsorted(OFF, j, side="right") - 1]))
parent = {}; info = {}
for b in range(1, 18):
    rate = M[f"{b}_rate"]; Eb = E[str(b)]; absE = np.abs(Eb); Cs = {k: M[f"{b}_{k[0]}_{k[1]}_C"] for k in VS}; Bs = {k: M[f"{b}_{k[0]}_{k[1]}_br"] for k in VS}
    etop = np.argsort(-absE, 1)[:, :40]
    for n in range(H[b]):
        rule = [int(OFF[p[0]] + p[1]) for p in mp[f"{b}_{n}"]["parents"]]
        edge = [int(j) for j in etop[n] if j not in rule][:3]; cands = rule + edge; best = (-1, None, None)
        for j in cands:
            rs, via = 0, ""
            for k in VS:
                C = float(Cs[k][n, j]); br = float(Bs[k][j]); L = C / max(br, 1e-6)
                if br > rate[n] and L > 1 and C * np.log(L) > rs: rs, via = C * np.log(L), f"{k[0]}-{k[1]} C {C:.2f} lift {L:.1f} ratio {br / max(rate[n], 1e-6):.1f}"
            er = int((absE[n] > absE[n, j]).sum()) + 1; sc = rs * np.log(1 + 1e4 / er)
            if sc > best[0]: best = (sc, j, via + f" edge rank {er} sign {'+' if Eb[n, j] > 0 else '-'}")
        parent[(b, n)] = loc(best[1]); info[f"{b}_{n}"] = {"parent": list(loc(best[1])), "score": round(best[0], 3), "why": best[2]}
    print("block", b, "done", flush=True)
# tree stats
kids = {}
for c, p in parent.items(): kids.setdefault(p, []).append(c)
def depth(c):
    d = 0
    while c in parent: c = parent[c]; d += 1
    return d, c
D = {c: depth(c) for c in parent}
stats = {"children_per_parent_median": float(np.median([len(v) for v in kids.values()])), "max_children": max(len(v) for v in kids.values()),
         "parents_with_children": len(kids), "chain_length_by_block": {b: float(np.median([D[(b, n)][0] for n in range(H[b])])) for b in range(1, 18)},
         "roots_used": len({D[c][1] for c in D}), "block0_neurons": H[0],
         "biggest_families": sorted([(len(v), list(p)) for p, v in kids.items()], reverse=True)[:15]}
print(json.dumps(stats, indent=0)[:1500])
json.dump({"neurons": info, "stats": stats}, open(os.path.join(ROOT, "research/clean/map/network_map_v2.json"), "w"), default=int)
