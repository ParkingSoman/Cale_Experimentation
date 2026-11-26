"""N11 gate (numpy only, light on the Mac): is B coarser than A? Inputs: read_A.npz and read_<B tag>.npz
(capture18.py, same 19,000 held + test photos, same order).

Per neuron (stage-4 block 1, and block 0):
  class profile = per-class mean of pos (max pre-ReLU input over positions), z-scored over the 1,000 classes
  breadth       = number of classes with z > 2 (how many kinds of thing its positive side covers)
  family        = the N10 WordNet family (results/read/families.json) of each of its top-10 classes
Matching: for each B neuron, the A neuron (same block) whose class profile correlates most with its own (over the
1,000 classes). Per-photo matching was dropped: most photos are noise for any one neuron.
Prints: breadth of all A and all B neurons (median, quartiles), and for B's car neurons (top car score): the
neuron, its matched A neuron and correlation, both top-class lists, both breadths, and the classes the B neuron
covers that its match does not.
Usage: python gate18.py B<k>aug_s<seed>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "coarse", "r18")
CI = json.load(open(os.path.join(HERE, "..", "results", "read", "imagenet_class_index.json")))
NAMES = [CI[str(i)][1] for i in range(1000)]
CARS = [407, 436, 468, 511, 609, 627, 656, 661, 717, 751, 817]


def prof(d, key):
    v = d[key].astype(np.float32)
    lab = d["lab"]
    cm = np.zeros((1000, v.shape[1]), np.float32)
    np.add.at(cm, lab, v)
    cm /= np.bincount(lab, minlength=1000)[:, None]
    return (cm - cm.mean(0)) / cm.std(0).clip(1e-6), v


def corr(a, b):
    a = (a - a.mean(0)) / a.std(0).clip(1e-6)
    b = (b - b.mean(0)) / b.std(0).clip(1e-6)
    return a.T @ b / len(a)


def main():
    tag = sys.argv[1]
    A = np.load(os.path.join(OUT, "read_A.npz"))
    B = np.load(os.path.join(OUT, f"read_{tag}.npz"))
    assert (A["ix"] == B["ix"]).all()
    res = {}
    for blk in (1, 0):
        pa, va = prof(A, f"pos{blk}")
        pb, vb = prof(B, f"pos{blk}")
        ba, bb = (pa > 2).sum(0), (pb > 2).sum(0)
        q = lambda x: [int(np.percentile(x, p)) for p in (25, 50, 75)]
        print(f"== block {blk}: breadth (classes with z > 2), quartiles: A {q(ba)} (n={len(ba)}), {tag} {q(bb)} (n={len(bb)})")
        C = corr(pb, pa)            # k_B x 512: correlation of the two class profiles over the 1,000 classes (same job)
        match = C.argmax(1)
        car_b = pb[CARS].mean(0)
        rows = []
        for j in np.argsort(-car_b)[:8]:
            i = match[j]
            tb = np.argsort(-pb[:, j])[:10]
            ta = np.argsort(-pa[:, i])[:10]
            extra = [NAMES[c] for c in tb if pa[c, i] < 1]
            rows.append(dict(b=int(j), a=int(i), corr=float(C[j, i]), breadth_b=int(bb[j]), breadth_a=int(ba[i]),
                             top_b=[NAMES[c] for c in tb], top_a=[NAMES[c] for c in ta], extra=extra))
            print(f"B n{j:3d} (car {car_b[j]:.2f}, breadth {bb[j]}) ~ A n{i:3d} (corr {C[j, i]:.2f}, breadth {ba[i]})")
            print(f"   B: {', '.join(NAMES[c] for c in tb)}")
            print(f"   A: {', '.join(NAMES[c] for c in ta)}")
            print(f"   B covers, A does not (A z < 1): {', '.join(extra) or '-'}")
        res[f"block{blk}"] = dict(breadth_A=q(ba), breadth_B=q(bb), best_corr_median=float(np.median(C.max(1))), cars=rows)
        print(f"   median best-match correlation of B neurons: {np.median(C.max(1)):.2f}")
    json.dump(res, open(os.path.join(OUT, f"gate_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
