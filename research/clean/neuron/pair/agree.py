"""Pair study, step 1: do the firing match and the writing match point to the same Base neuron?

Firing match: correlation of per-photo peaks (tools.py, tools_peakR_b*.npy).
Writing match: cosine of direct class-score effects (writes.py, writesR_b*.npy).
For each Small neuron: is the Base argmax the same under both? Chance is 1 / 4096.
Also: a joint score (rank-combined) and its one-to-one assignment, saved for the next steps.
Null for the writing match: Base vectors with the 1,000 classes permuted.
"""
import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

from cache import NB, OUT


def main():
    res = {}
    V = np.load(os.path.join(OUT, "writes_V.npz"))
    rng = np.random.default_rng(0)
    for b in range(NB):
        Rf = np.load(os.path.join(OUT, f"tools_peakR_b{b}.npy")).astype(np.float32)
        Rw = np.load(os.path.join(OUT, f"writesR_b{b}.npy")).astype(np.float32)
        af, aw = Rf.argmax(1), Rw.argmax(1)
        same = af == aw
        # top-5 agreement: firing argmax among the 5 best writing matches
        top5w = np.argsort(-Rw, 1)[:, :5]
        in5 = (top5w == af[:, None]).any(1)
        # null for writes: permute classes in Base
        vs = V[f"s{b}"]
        vb = V[f"b{b}"][:, rng.permutation(1000)]
        vs = vs / np.linalg.norm(vs, axis=1, keepdims=True)
        vb = vb / np.linalg.norm(vb, axis=1, keepdims=True)
        null_best = (vs @ vb.T).max(1).mean()
        # joint score: sum of the two, each z-scored per Small neuron
        z = lambda R: (R - R.mean(1, keepdims=True)) / R.std(1, keepdims=True)
        J = z(Rf) + z(Rw)
        r, c = linear_sum_assignment(-J)
        hung_f, hung_w = Rf[r, c], Rw[r, c]
        np.save(os.path.join(OUT, f"match_b{b}.npy"), np.stack([r, c]))
        res[f"b{b}"] = dict(argmax_same=float(same.mean()), fire_argmax_in_write_top5=float(in5.mean()),
                            chance=1 / Rf.shape[1], write_null_best_mean=float(null_best),
                            fire_of_agreeing=float(Rf.max(1)[same].mean()) if same.any() else None,
                            write_of_agreeing=float(Rw.max(1)[same].mean()) if same.any() else None,
                            joint_hung_fire_median=float(np.median(hung_f)),
                            joint_hung_write_median=float(np.median(hung_w)),
                            joint_both_over_05=float(((hung_f > .5) & (hung_w > .5)).mean()))
        print(b, res[f"b{b}"], flush=True)
    json.dump(res, open(os.path.join(OUT, "agree.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
