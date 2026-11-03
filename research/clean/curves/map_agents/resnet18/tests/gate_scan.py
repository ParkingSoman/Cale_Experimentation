# serves: H-map
"""Scan stages 2,3 for gate candidates: channel fires (hx) on cars and has most-negative z (zn) on dogs, or the reverse.
Uses stored per-photo extremes of the reading photos only."""
import numpy as np
R = "cache/map18/read/"
lab = np.load(R + "labels.npy")
cars = np.isin(lab, [436, 511, 627, 751, 817, 468, 609, 656, 717, 479, 581, 407, 864, 675, 734])
dogs = (lab >= 151) & (lab <= 268)
print("n cars", cars.sum(), "n dogs", dogs.sum())
for s in (2, 3):
    hx = np.asarray(np.load(R + f"s{s}b1_hx.npy", mmap_mode="r"), dtype=np.float32)
    zn = np.asarray(np.load(R + f"s{s}b1_zn.npy", mmap_mode="r"), dtype=np.float32)
    if s == 2:
        print("zn min/max", zn.min(), zn.max(), "hx", hx.min(), hx.max())
    out = []
    for c in range(hx.shape[1]):
        carH, dogH = hx[cars, c].mean(), hx[dogs, c].mean()
        carN, dogN = zn[cars, c].mean(), zn[dogs, c].mean()
        sd = hx[:, c].std() + 1e-6
        sn = zn[:, c].std() + 1e-6
        g = (carH - dogH) / sd + (carN - dogN) / sn
        out.append((g, c, carH, dogH, carN, dogN))
    out.sort(reverse=True)
    print(f"stage {s} gate-like (car high, dog negative)")
    for o in out[:10]:
        print("  s%d:%d g=%.2f carH=%.2f dogH=%.2f carZn=%.2f dogZn=%.2f" % (s, o[1], o[0], o[2], o[3], o[4], o[5]))
    print(f"stage {s} reverse (dog high, car negative)")
    for o in out[-6:]:
        print("  s%d:%d g=%.2f carH=%.2f dogH=%.2f carZn=%.2f dogZn=%.2f" % (s, o[1], o[0], o[2], o[3], o[4], o[5]))
# note: result 2026-11-03 in NOTES.md (gate lines)
