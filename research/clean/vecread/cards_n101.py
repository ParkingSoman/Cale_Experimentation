"""Step 6 (runs only after vr_check K1-K6 pass): read n101's vectors against the simple average fold's vectors.

Vectors (all in A's spaces: B128aug_s1 shares A's stages 1-3, skip and head, checked 2026-11-23, difference 0.0):
  r101   B W1_0[101] x bn1 scale                 read-like, s3k (256 x 3 x 3 kernel, flattened)
  rFold  a-weighted mean of the 11 children's    read-like, s3 kernel (full fold of oc_nuudle3.py)
         A W1_0 rows x bn1 scale                 read-like, s3k
  w101   B W2_0[:, 101] x B bn2 scale, 3x3 sum   write-like, s4b0 per cell 512
  wFold  sum of the 11 children's A W2_0 columns write-like, s4b0 per cell 512
         x A bn2 scale, 3x3 sum
Children and a-weights: onecoarse/results/oc_children_w.npz (order[:11], score = channel-69 read x write norm).
Output: results/n101_vs_fold.json and .md
"""
import json
import os

import numpy as np

import vecread as vr

CLEAN = os.path.dirname(vr.HERE)
R18 = os.path.join(CLEAN, "neuron", "results", "coarse", "r18")
K = 11


def bn_scale(bn):
    g, _, _, v = bn
    return g / np.sqrt(v + 1e-5)


def vectors():
    A = np.load(os.path.join(R18, "w_A.npz"))
    B = np.load(os.path.join(R18, "w_B128aug_s1.npz"))
    cw = np.load(os.path.join(CLEAN, "onecoarse", "results", "oc_children_w.npz"))
    kids = cw["order"][:K]
    a = cw["score"][kids]
    RA = A["W1_0"] * bn_scale(A["bn1_0"])[:, None, None, None]
    RB = B["W1_0"] * bn_scale(B["bn1_0"])[:, None, None, None]
    WA = A["W2_0"] * bn_scale(A["bn2_0"])[:, None, None, None]
    WB = B["W2_0"] * bn_scale(B["bn2_0"])[:, None, None, None]
    v = {"r101": RB[101], "rFold": (a[:, None, None, None] * RA[kids]).sum(0) / a.sum(),
         "w101": WB[:, 101].sum((1, 2)), "wFold": WA[:, kids].sum((1, 2)).sum(1)}
    spread = {"w101": WB[:, 101], "wFold": WA[:, kids].sum(1)}
    centre = {k: float((s[:, 1, 1] ** 2).sum() / (s ** 2).sum()) for k, s in spread.items()}
    return v, centre, [int(k) for k in kids]


def main():
    v, centre, kids = vectors()
    held = vr.Held()
    out = {"children": kids, "write_centre_share": centre}
    d3 = vr.Dictionary("s3k")
    out["read"] = vr.compare(d3, v["r101"].reshape(-1), v["rFold"].reshape(-1), kind="read", held=held)
    d4 = vr.Dictionary("s4b0")
    out["write"] = vr.compare(d4, v["w101"], v["wFold"], kind="write", held=held)
    json.dump(out, open(os.path.join(vr.RES, "n101_vs_fold.json"), "w"), indent=1, default=str)
    md = [vr.card_md("Read: r101 (a) against rFold (b)", out["read"]),
          vr.card_md("Write: w101 (a) against wFold (b)", out["write"]),
          f"Write centre share (3 x 3): {centre}", f"Children: {kids}"]
    open(os.path.join(vr.RES, "n101_vs_fold.md"), "w").write("\n\n".join(md))
    print("\n\n".join(md))


if __name__ == "__main__":
    main()
