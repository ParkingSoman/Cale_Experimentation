"""Phase 5 scout: which 3b neurons are circle detectors? Centre response to centred full circles (radius 6-60 px)
vs best response to single arcs (any orientation). Circle neurons: strong on circles, much weaker on arcs."""
import numpy as np, torch
from inception import *
from stimuli import arc, circle
m = load_model(); R = [6, 10, 15, 20, 30, 40, 60]
with torch.no_grad():
    zc = F.relu(pre_3b(m, to_3a(m, prep(np.stack([circle(radius=r) for r in R])))))[:, :, 13:16, 13:16].amax((2, 3)).cpu().numpy()   # (7,480)
    za = np.stack([F.relu(pre_3b(m, to_3a(m, prep(np.stack([arc(t, radius=r) for t in range(0, 360, 30)])))))[:, :, 11:18, 11:18].amax((0, 2, 3)).cpu().numpy() for r in R])
best_c = zc.max(0); best_a = za.max(0); ratio = best_c / (best_a + 1)
order = np.argsort(-(best_c * np.clip(ratio - 1, 0, None)))
br = lambda i: "1x1" if i < 128 else "3x3" if i < 320 else "5x5" if i < 416 else "pool"
print(" n   branch circle_max arc_max ratio  pref_radius  circle response by radius")
for i in order[:20]:
    print(f"{i:4d} {br(i):5s} {best_c[i]:9.0f} {best_a[i]:7.0f} {ratio[i]:5.2f}  {R[zc[:, i].argmax()]:5d}   {np.round(zc[:, i]/100).astype(int)}")
print("curve neurons for comparison:", [(n, round(float(ratio[n]), 2)) for n in CURVES])
