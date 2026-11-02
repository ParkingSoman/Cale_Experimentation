"""For each anchor: best containing parent among ALL neurons in blocks 13..(b-1) that are at least 2x broader and on for at most 25% of images.
Same on-rule as last_t7_contain.py. Prints top 4 by C (containment) with breadth and lift; null = median C."""
import numpy as np
from last_lib import *
d = cache(); ON = {}
for b in range(13, 17):
    M = d[f"max{b}"][:].astype(np.float32); ON[b] = M > .5 * np.percentile(M, 99.9, axis=0)
v17 = d["max17"][:, 2283].astype(np.float32); ch17 = v17 > .5 * np.percentile(v17, 99.9)
for b, n in [(15, 16), (16, 384), (16, 26), (16, 2062), (17, 2283)]:
    ch = ON[b][:, n] if b < 17 else ch17; br = ch.mean(); rows = []
    for pb in range(13, b):
        O = ON[pb]; p = O.mean(0); C = O[ch].mean(0); ok = (p >= 2 * br) & (p <= .25)
        for j in np.where(ok)[0]: rows.append((C[j], pb, j, p[j]))
    rows.sort(reverse=True); allC = np.array([r[0] for r in rows])
    print(f"{b} n{n} breadth {br:.4f}; candidates {len(rows)}, median C {np.median(allC):.2f}, 99th pct C {np.percentile(allC,99):.2f}")
    for C, pb, j, p in rows[:4]: print(f"    {pb} n{j:<5d} C {C:.2f} breadth {p:.3f} lift {C/p:.1f}  top classes {[c for c,_ in top_classes(amax(pb,j),6)]}")
