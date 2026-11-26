# scale: neurons=s3:69, A:children; net=A; photos=saved; where=mac
"""Read oc_dial3.npz (LOG "Step 3: the exact route cut, on the logit change"). Mac numpy.
Shares are sums over photos of dot products of centred logit changes. T = Ro + O exactly.
L1: weak range sum(Ro.T)/sum(T.T) < .30. L2: sum(M.Ro)/sum(Ro.Ro) >= .7. L3: sum(M.O)/sum(O.O) > .10.
Also: fraction of the total change the morph restores, sum(M.T)/sum(T.T), and the leftover after the morph,
1 - |T - M|^2 / |T|^2 (logit-space recovery)."""
import numpy as np

d = np.load("results/oc_dial3.npz")
pk = d["dev_peak"]
g = {k[3:]: d[k] for k in d.files if k.startswith("s3_")}
print(f"additivity error max {g['err'].max():.1e}")
BINS = [(0, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 2.92), (2.92, 99), (.5, 2)]
print("bin         n   route share of T | morph on route (M.Ro/Ro.Ro) | morph on other (M.O/O.O) | M.T/T.T | logit recovery 1-|T-M|^2/|T|^2")
for lo, hi in BINS:
    m = (pk >= lo) & (pk < hi)
    s = {k: v[m].sum() for k, v in g.items()}
    rec = 1 - (s["TT"] - 2 * s["MT"] + s["MM"]) / s["TT"]
    tag = "  <- weak range" if (lo, hi) == (.5, 2) else ""
    print(f"{lo:4.2f}-{hi:5.2f} {m.sum():4d}   {s['RoT'] / s['TT']:.2f}              | {s['MRo'] / s['RoRo']:.2f}                        | "
          f"{s['MO'] / s['OO']:.2f}                     | {s['MT'] / s['TT']:.2f}    | {rec:.2f}{tag}")
w = (pk >= .5) & (pk < 2)
s = {k: v[w].sum() for k, v in g.items()}
print(f"\nL1 route share {s['RoT'] / s['TT']:.2f} -> {'PASS' if s['RoT'] / s['TT'] < .3 else 'FAIL'}")
print(f"L2 morph on route {s['MRo'] / s['RoRo']:.2f} -> {'PASS' if s['MRo'] / s['RoRo'] >= .7 else 'FAIL'}")
print(f"L3 morph on other {s['MO'] / s['OO']:.2f} -> {'PASS' if s['MO'] / s['OO'] > .1 else 'FAIL'}")
roo = s["RoT"] - s["RoRo"]
print(f"Ro.O / sqrt(RoRo OO) = {roo / np.sqrt(s['RoRo'] * s['OO']):.2f}")
print(f"|M|^2 / |Ro|^2 = {s['MM'] / s['RoRo']:.2f}; cos(M, Ro) = {s['MRo'] / np.sqrt(s['MM'] * s['RoRo']):.2f}; "
      f"cos(M, O) = {s['MO'] / np.sqrt(s['MM'] * s['OO']):.2f}; cos(M, T) = {s['MT'] / np.sqrt(s['MM'] * s['TT']):.2f}")
