"""Whole-network coarse-to-fine map: parent of every neuron in blocks 1-17 by score = C * log(lift) (best of
view/point x pos/neg; parent broader than the child). Confident if the score beats the 95th percentile of the
best-score of shuffled-weight null readouts in the same block (m3_null.npz). Output research/clean/map/network_map.json."""
import json, os
import numpy as np
from convnext import *
M = np.load(os.path.join(HF, "imagenet", "m3_matrices.npz")); N = np.load(os.path.join(HF, "imagenet", "m3_null.npz"))
H = [384]*3 + [768]*3 + [1536]*9 + [3072]*3; OFF = np.concatenate([[0], np.cumsum(H)]); VS = [(v, s) for v in ("view", "point") for s in ("pos", "neg")]
loc = lambda j: (int(np.searchsorted(OFF, j, side="right") - 1), int(j - OFF[np.searchsorted(OFF, j, side="right") - 1]))


def scores(Cget, rate, b):
    best = None; arg = None
    for k, (v, s) in enumerate(VS):
        C = Cget(v, s).astype(np.float32); br = M[f"{b}_{v}_{s}_br"]; L = C / np.maximum(br[None], 1e-6)
        sc = np.where((br[None] > rate[:, None]) & (L > 1), C * np.log(np.maximum(L, 1)), 0)
        if best is None: best = sc; arg = np.zeros(sc.shape, np.int8)
        else: upd = sc > best; best = np.where(upd, sc, best); arg = np.where(upd, k, arg)
    return best, arg


out = {"neurons": {}, "blocks": {}}
for b in range(1, 18):
    rate = M[f"{b}_rate"]; S, A = scores(lambda v, s: M[f"{b}_{v}_{s}_C"], rate, b)
    nS, _ = scores(lambda v, s: N[f"{b}_{v}_{s}_nC"], N[f"{b}_nrate"], b); thr = float(np.quantile(nS.max(1), .95))
    top = np.argsort(-S, 1)[:, :3]; conf = S.max(1) > thr
    for n in range(H[b]):
        j = int(top[n, 0]); v, s = VS[A[n, j]]; C = float(M[f"{b}_{v}_{s}_C"][n, j]); br = float(M[f"{b}_{v}_{s}_br"][j])
        out["neurons"][f"{b}_{n}"] = {"on_rate": float(rate[n]), "parents": [list(loc(int(t))) + [round(float(S[n, t]), 3)] for t in top[n]],
                                      "via": f"{v}-{s}", "C": round(C, 3), "lift": round(C / max(br, 1e-6), 2),
                                      "breadth_ratio": round(br / max(rate[n], 1e-6), 2), "confident": bool(conf[n])}
    pb = np.array([loc(int(t))[0] for t in top[:, 0]])
    ratio = np.array([out["neurons"][f"{b}_{n}"]["breadth_ratio"] for n in range(H[b])])
    out["blocks"][b] = {"null_thr": thr, "confident": float(conf.mean()), "null_rate_above_thr": float((nS.max(1) > thr).mean()),
                        "median_score": float(np.median(S.max(1))), "parent_dist_median": float(np.median(b - pb[conf])) if conf.any() else None,
                        "twin_share_of_confident": float((ratio[conf] < 1.5).mean()) if conf.any() else None,
                        "median_breadth_ratio_confident": float(np.median(ratio[conf])) if conf.any() else None}
    o = out["blocks"][b]
    print(f"block {b:2d}: confident {o['confident']:.2f} (null thr {thr:.2f}) | parent {o['parent_dist_median']} blocks back | twins {o['twin_share_of_confident']} | breadth ratio {o['median_breadth_ratio_confident']}")
w = np.array(H[1:]); tot = sum(out["blocks"][b]["confident"] * H[b] for b in range(1, 18)) / w.sum()
out["whole_confident"] = float(tot); print(f"WHOLE NETWORK blocks 1-17: confident parent {tot:.3f}")
os.makedirs(os.path.join(ROOT, "research/clean/map"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "research/clean/map/network_map.json"), "w"), default=float)
