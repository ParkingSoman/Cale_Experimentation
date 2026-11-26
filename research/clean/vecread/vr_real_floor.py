"""Real-vector reference (K5b): readings of all of network A's own vectors in a space, so a reading can be placed as a
percentile among real vectors, not only against random ones.
  writes: s4b0, A's 512 block-0 write columns (3 x 3 summed, bn-scaled), raw metric
  reads:  s3k,  A's 512 block-0 read kernels (bn-scaled), whitened pattern, hub z
For each: spread of the top score and the leftover; how often each entry is top-1 / in the top 5 (hubs among real
vectors); spread of the top data-side IoU (with random vectors for comparison). Then n101 and the fold as percentiles.
Output: results/vr_real_floor.json
"""
import json
import os
from collections import Counter

import numpy as np

import vecread as vr
from cards_n101 import R18, bn_scale, vectors


def spread(x):
    return {q: round(float(np.percentile(x, p)), 3) for q, p in (("5th", 5), ("median", 50), ("95th", 95))}


def pct(x, v):
    return round(float((np.asarray(x) < v).mean() * 100), 1)


def readings(dic, held, V, kind):
    tops, lefts, top1, top5, ious = [], [], [], [], []
    for v in V:
        r = vr.read_vector(dic, v, kind, held=None, top=5)
        tops.append(r["up"][0][2])
        lefts.append(r["leftover"])
        top1.append(r["up"][0][0])
        top5 += [x[0] for x in r["up"]]
    vals = held.project(dic.space, np.asarray(V))
    for j in range(vals.shape[2]):
        ious.append(vr.examples(held, dic, vals[:, :, j], top=1)[0][2])
    return np.array(tops), np.array(lefts), top1, top5, np.array(ious)


def main():
    A = np.load(os.path.join(R18, "w_A.npz"))
    held = vr.Held()
    rng = np.random.default_rng(5)
    v, _, _ = vectors()
    sets = {
        "write_s4b0": ("s4b0", "write", (A["W2_0"] * bn_scale(A["bn2_0"])[:, None, None, None]).sum((2, 3)).T,
                       {"w101": v["w101"], "wFold": v["wFold"]}),
        "read_s3k": ("s3k", "read", (A["W1_0"] * bn_scale(A["bn1_0"])[:, None, None, None]).reshape(512, -1),
                     {"r101": v["r101"].reshape(-1), "rFold": v["rFold"].reshape(-1)}),
    }
    out = {}
    for name, (space, kind, V, probes) in sets.items():
        dic = vr.Dictionary(space)
        tops, lefts, top1, top5, ious = readings(dic, held, V, kind)
        R = rng.standard_normal((200, V.shape[1]))
        _, _, _, _, ious_rand = readings(dic, held, R, kind)
        o = {"n_real": len(V), "top_score_real": spread(tops), "leftover_real": spread(lefts),
             "top_iou_real": spread(ious), "top_iou_random": spread(ious_rand),
             "hubs_top1": [(str(k), round(c / len(V), 3)) for k, c in Counter(top1).most_common(6)],
             "hubs_top5": [(str(k), round(c / len(V), 3)) for k, c in Counter(top5).most_common(6)],
             "uniform_share": round(1 / dic.kept.sum(), 4)}
        for pn, pv in probes.items():
            r = vr.read_vector(dic, pv, kind, held=held, top=5)
            o[pn] = {"top": r["up"][0], "top_score_pct_among_real": pct(tops, r["up"][0][2]),
                     "leftover": r["leftover"], "leftover_pct_among_real": pct(lefts, r["leftover"]),
                     "top_iou": r["examples"][0], "top_iou_pct_among_real": pct(ious, r["examples"][0][2])}
        out[name] = o
        print(name, json.dumps(o, indent=1, default=str), flush=True)
    json.dump(out, open(os.path.join(vr.RES, "vr_real_floor.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
