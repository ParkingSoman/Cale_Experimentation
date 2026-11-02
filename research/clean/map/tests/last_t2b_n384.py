"""16 n384 follow-up: keep ONLY the band where the neuron peaks (rest gray) vs keep everything except it, on half-track images.
Peak row band = the 7x7 row of the max +-1 row (32 px per row)."""
import numpy as np
from last_lib import *
B, N = 16, 384
X, y, ana = data(); nm = names(); ya = y[ana]
for c in ("half track", "tank"):
    ix = ana[ya == nm.index(c)]; im = np.stack([np.asarray(X[i]) for i in ix]); Z = zmaps(B, N, im); base = Z.max((1, 2)); ok = base > 2
    keep, drop, keepcell = im.copy(), im.copy(), im.copy()
    for j in range(len(im)):
        r, cc = np.unravel_index(Z[j].argmax(), (7, 7)); t, b_ = max(0, (r - 1) * 32), min(224, (r + 2) * 32)
        keep[j, :t] = 128; keep[j, b_:] = 128; drop[j, t:b_] = 128
        l, rr = max(0, (cc - 1) * 32), min(224, (cc + 2) * 32); keepcell[j] = 128; keepcell[j, t:b_, l:rr] = im[j, t:b_, l:rr]
    for name, e in (("keep peak row band only", keep), ("gray out peak row band", drop), ("keep 3x3-cell window at peak only", keepcell)):
        r = zmaps(B, N, e).max((1, 2)); print(c, name, f"median ratio {np.median(r[ok]/base[ok]):.2f} (n={ok.sum()})")
    rows = [np.unravel_index(Z[j].argmax(), (7, 7))[0] for j in np.where(ok)[0]]; print(c, "peak rows (0=top,6=bottom):", np.bincount(rows, minlength=7))
