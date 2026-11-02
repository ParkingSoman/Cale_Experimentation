"""At the coarse neuron's strongest spots: how many fine members are on, (a) at q, (b) moved to q - r u_theta.
Also the members' mean pre-activation there. Compared with all spots."""
import sys, numpy as np, torch
from inception import *
from family import Family
from s7_lib import load_maps, moved
tag = sys.argv[1]; r = float(sys.argv[2]) if len(sys.argv) > 2 else 2
Zp, C = load_maps(tag, pre=True)                        # fine pre-activations (N,10,28,28), coarse pre (N,28,28)
M = (slice(None), slice(4, 24), slice(4, 24))
c = C[M].ravel(); top = c >= np.quantile(c, 0.995); hi = c >= np.quantile(c, 0.95)
at_q = Zp[:, :, 4:24, 4:24].transpose(0, 2, 3, 1).reshape(-1, 10)
mv = moved(Zp, r)[:, :, 4:24, 4:24].transpose(0, 2, 3, 1).reshape(-1, 10)
np.set_printoptions(precision=2, suppress=True, linewidth=160)
for nm, A in [("at q", at_q), (f"moved r={r}", mv)]:
    for lab, sel in [("all spots", slice(None)), ("coarse top 5%", hi), ("coarse top 0.5%", top)]:
        n = (A[sel] > 0).sum(1)
        print(f"{nm:12s} {lab:16s}: members on mean {n.mean():.2f}  dist {np.bincount(n, minlength=11)[:11] / len(n)}  mean pre/100 {A[sel].mean(0)/100}")
print("coarse firing rate:", (c > 0).mean())
