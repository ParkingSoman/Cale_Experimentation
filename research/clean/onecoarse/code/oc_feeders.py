"""Step-2 feeder tables from oc_split.npz: per neuron and side, the feeders that carry the value at the cell,
averaged over the 30 band photos (10 per band), with how many of the 30 photos have that feeder in their own top 5.
Feeders: stage-3 channels s3:i for the children and n101; layer3[1] inner neurons l3b1:k and the skip for the parent."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, "..", "results", "oc_split.npz"))


def name(n, i):
    if n == "p":
        return "skip" if i == 256 else f"l3b1:{i}"
    return f"s3:{i}"


for n in (sys.argv[1:] or ["p", "k305", "k97", "k23", "k297", "k108", "k163", "c101"]):
    for side in ("pos", "neg"):
        sp, tot = D[f"{n}_split_{side}"], D[f"{n}_tot_{side}"]
        m = sp.mean(0)
        sign = 1 if side == "pos" else -1
        o = np.argsort(-sign * m)
        top5 = np.argsort(-sign * sp, 1)[:, :5]
        print(f"\n{n} {side}: value - offset at the cell, mean {tot.mean():+.2f}; sum of all feeders = that value")
        for i in o[:8]:
            print(f"   {name(n, i):10s} {m[i]:+.3f}   in the photo's own top 5: {int((top5 == i).any(1).sum())}/30")
        opp = np.argsort(sign * m)[:4]
        print("   strongest against:", [(name(n, i), round(float(m[i]), 3)) for i in opp])
