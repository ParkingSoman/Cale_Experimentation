"""Children check: for each anchor and its top readers (S2 'down'), is the reader's on-region inside the anchor's?
C_child = P(anchor on | reader on); breadth of each. Same on-rule as last_t7_contain.py."""
import numpy as np
from last_lib import *


def on(b, n): a = amax(b, n); return a > .5 * np.percentile(a, 99.9)


R = {(15, 16): [(16, 1328), (17, 2911), (16, 1738)], (16, 384): [(17, 1479), (17, 689), (17, 651)],
     (16, 26): [(17, 2795), (17, 407), (17, 1871)], (16, 2062): [(17, 542), (17, 812), (17, 226)]}
for (b, n), rs in R.items():
    a = on(b, n); print(f"anchor {b} n{n} breadth {a.mean():.4f}")
    for rb, rn in rs:
        r = on(rb, rn); print(f"   reader {rb} n{rn:<5d} breadth {r.mean():.4f}  P(anchor on | reader on) {a[r].mean():.2f}  P(reader on | anchor on) {r[a].mean():.2f}  top {[c for c,_ in top_classes(amax(rb,rn),5)]}")
