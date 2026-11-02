"""S5: score candidate parent rules against the hand answer key (S4 cards). Rank of the hand parent among ALL earlier
neurons (1 = best). Rules use only M3 containment matrices and E1 edges (no hand input)."""
import json, os
import numpy as np
from convnext import *
M = np.load(os.path.join(HF, "imagenet", "m3_matrices.npz")); E = np.load(os.path.join(HF, "imagenet", "e1_edges.npz"))
H = [384]*3 + [768]*3 + [1536]*9 + [3072]*3; OFF = np.concatenate([[0], np.cumsum(H)])
KEY = {(1, 351): [(0, 245)], (3, 623): [(2, 61)], (4, 499): [(1, 365)], (5, 135): [(4, 212)], (8, 4): [(6, 205)],
       (10, 1316): [(9, 686)], (11, 1031): [(10, 1172)], (12, 136): [(9, 1137)], (14, 1326): [(13, 78)], (14, 33): [(12, 548)],
       (14, 831): [(3, 92)], (15, 16): [(14, 1138)], (16, 384): [(15, 1162), (13, 212), (14, 103)], (16, 26): [(15, 2142), (14, 628)],
       (16, 2062): [(13, 1136)], (17, 2283): [(16, 2886), (14, 1072)]}
g = lambda b, n: int(OFF[b] + n)
def rules(b, n):
    rate = M[f"{b}_rate"][n]; S = {}
    Cs = {f"{v}_{s}": M[f"{b}_{v}_{s}_C"][n].astype(np.float32) for v in ("view", "point") for s in ("pos", "neg")}
    Bs = {k: M[f"{b}_{k}_br"] for k in Cs}
    e = E[str(b)][n]
    lifts = {k: Cs[k] / np.maximum(Bs[k], 1e-6) for k in Cs}
    broader = {k: Bs[k] > rate for k in Cs}
    # R1 edge (positive direct input), R2 |edge|
    S["edge+"] = e; S["|edge|"] = np.abs(e)
    # R3 containment (best variant), R4 lift, R5 C*log(lift) with broader, R6 = R5 restricted to top-100 |edge|
    C = np.max(np.stack(list(Cs.values())), 0); L = np.max(np.stack([np.where(broader[k], lifts[k], 0) for k in Cs]), 0)
    S["containment"] = C; S["lift"] = L
    sc = np.max(np.stack([np.where(broader[k] & (lifts[k] > 1), Cs[k] * np.log(np.maximum(lifts[k], 1)), 0) for k in Cs]), 0)
    S["C*log(lift)"] = sc
    top = np.argsort(-np.abs(e))[:100]; m = np.full_like(sc, -1); m[top] = sc[top]; S["C*log(lift) in top-100 |edge|"] = m
    top = np.argsort(-np.abs(e))[:30]; m = np.full_like(sc, -1); m[top] = sc[top]; S["C*log(lift) in top-30 |edge|"] = m
    return S
res = {}; names = None
for (b, n), parents in KEY.items():
    S = rules(b, n); names = list(S)
    res[f"{b}_{n}"] = {k: int(min((v > v[g(pb, pn)]).sum() + 1 for pb, pn in parents)) for k, v in S.items()}
print("rule | median rank | rank 1 | top 5 | top 20 | ranks per child")
out = {}
for k in names:
    r = np.array([res[c][k] for c in res]); out[k] = {"median": float(np.median(r)), "top1": int((r == 1).sum()), "top5": int((r <= 5).sum()), "top20": int((r <= 20).sum())}
    print(f"{k:32s} | {np.median(r):6.0f} | {(r==1).sum():2d} | {(r<=5).sum():2d} | {(r<=20).sum():2d} | {list(r)}")
json.dump({"per_child": res, "summary": out}, open(os.path.join(RES, "vision-s5-rules.json"), "w"), indent=1)
