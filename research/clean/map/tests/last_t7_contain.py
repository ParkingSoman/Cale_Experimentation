"""Territory containment of each anchor inside its parent candidates (20k analysis images, image-level).
on = image max > .5 x the neuron's 99.9th-percentile max. C = P(parent on | child on); breadth = P(parent on);
lift = C / breadth; back = P(child on | parent on). Parent = broader (breadth > child's) and contains (C high)."""
import numpy as np
from last_lib import *
P = {(15, 16): [(14, 1138), (14, 402), (13, 811), (14, 1015), (14, 1526)],
     (16, 384): [(15, 1162), (13, 212), (14, 103), (15, 2108), (15, 1424)],
     (16, 26): [(15, 2722), (15, 2142), (15, 152), (15, 1134), (15, 1044)],
     (16, 2062): [(15, 2065), (15, 2584), (15, 3005), (15, 1019), (15, 190), (15, 2560)],
     (17, 2283): [(16, 2886), (16, 2558), (15, 1331), (15, 1019), (16, 2904), (16, 2117), (15, 72)]}


def on(b, n): a = amax(b, n); return a > .5 * np.percentile(a, 99.9)


for (b, n), cs in P.items():
    ch = on(b, n); print(f"child {b} n{n}: breadth {ch.mean():.4f} ({ch.sum()} imgs)")
    for pb, pn in cs:
        p = on(pb, pn); C = p[ch].mean(); print(f"   {pb} n{pn:<5d} breadth {p.mean():.4f}  C {C:.2f}  lift {C/max(p.mean(),1e-9):5.1f}  back {ch[p].mean():.2f}")
