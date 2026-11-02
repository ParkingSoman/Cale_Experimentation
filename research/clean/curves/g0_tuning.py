"""Gate 0: radial tuning of the ten curve neurons (3b, centre position), on arcs at 36 orientations and 4 radii."""
import json, numpy as np, torch
from inception import *
from stimuli import arc, line
m = load_model(); TH = np.arange(0, 360, 10)
out = {}
with torch.no_grad():
    for r in [20, 40, 60, 100]:
        S = np.stack([arc(t, radius=r) for t in TH])
        h = F.relu(pre_3b(m, to_3a(m, prep(S))))   # (36, 480, 28, 28)
        c = h[:, :, 12:17, 12:17].amax((2, 3)).cpu().numpy()   # max over the centre 5x5 positions
        out[r] = c
    L = np.stack([line(t) for t in TH[:18]]); hl = F.relu(pre_3b(m, to_3a(m, prep(L))))[:, :, 12:17, 12:17].amax((2, 3)).cpu().numpy()
allc = np.concatenate([out[r] for r in out])
print("neuron  pref_deg(r=40)  peak(r=40)  sharpness(peak/mean)  best_r  line_max/arc_max  rank_among_480_on_arcs")
mx_all = allc.max(0)
order = np.argsort(-mx_all)
for n in CURVES:
    t = out[40][:, n]; best_r = max(out, key=lambda r: out[r][:, n].max())
    print(f"{n:5d} {TH[t.argmax()]:8d} {t.max():10.1f} {t.max()/(t.mean()+1e-6):10.2f} {best_r:8d} {hl[:, n].max()/(mx_all[n]+1e-6):10.2f} {int(np.where(order==n)[0][0]):6d}")
print("top-20 neurons on arcs (any r):", order[:20].tolist())
np.savez("results_g0_tuning.npz", **{str(r): out[r] for r in out}, line=hl, th=TH)
