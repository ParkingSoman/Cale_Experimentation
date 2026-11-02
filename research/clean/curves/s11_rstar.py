"""r* from the large model and images alone: co-firing of the ten fine curve members after moving each by r toward
its centre of curvature. Measures per r: mean pairwise post-ReLU correlation, P(>=3 members on | >=1 on), and the
mean number on where any is on. Reversed direction as control. 1000 train images, inner positions."""
import numpy as np, torch
from inception import *
from s7_lib import moved, PREF
m = load_model(); X, y = load_images(); tr, dv, se = split(y); fam = list(np.load("cache/s1_fine.npz")["fam"])
idx = np.sort(np.random.default_rng(5).choice(tr, 1000, replace=False)); Zs = []
with torch.no_grad():
    for i in range(0, len(idx), 100):
        Zs.append(F.relu(pre_3b(m, to_3a(m, prep(X[idx[i:i+100]])))[:, fam]).cpu().numpy().astype(np.float32))
Z = np.concatenate(Zs)
for r in [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, -1, -2, -3]:
    A = moved(Z, r)[:, :, 5:23, 5:23].transpose(0, 2, 3, 1).reshape(-1, 10)
    Cm = np.corrcoef(A.T); pc = (Cm.sum() - 10) / 90; on = (A > 0).sum(1)
    print(f"r={r:+.1f} ({8*r:+.0f}px): mean pairwise corr {pc:+.3f}  P(>=3 on | >=1 on) {(on >= 3).sum()/(on >= 1).sum():.3f}  "
          f"P(>=5 on | >=1) {(on >= 5).sum()/(on >= 1).sum():.4f}  sum-of-members CV {A.sum(1).std()/A.sum(1).mean():.2f}")
