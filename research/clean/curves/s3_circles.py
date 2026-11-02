"""Coarse vs fine family on centred full circles and on single arcs, by radius (centre position response, raw)."""
import sys, numpy as np, torch
from inception import *
from family import Family
from stimuli import arc, circle
m = load_model(); S1 = np.load("cache/s1_fine.npz"); fam = list(S1["fam"])
ck = torch.load(f"cache/ckpt/{sys.argv[1]}.pt", map_location="cpu", weights_only=False); fm = Family(m, ck["fam"], ck["k"], ks=ck.get("ks", 5)).to(DEV); fm.load_state_dict(ck["state"])
R = [6, 10, 15, 20, 30, 40, 60]
with torch.no_grad():
    C = np.stack([circle(radius=r) for r in R]); ins = inputs_3b(m, to_3a(m, prep(C)))
    zc = fm.pre(ins)[:, 0].cpu().numpy(); zf = pre_3b(m, ins[0])[:, fam].cpu().numpy()
    A = np.stack([arc(t, radius=r) for r in R for t in range(0, 360, 30)]); ia = inputs_3b(m, to_3a(m, prep(A)))
    ac = fm.pre(ia)[:, 0].cpu().numpy().reshape(len(R), 12, 28, 28); af = F.relu(pre_3b(m, ia[0])[:, fam]).cpu().numpy().reshape(len(R), 12, 10, 28, 28)
for i, r in enumerate(R):
    print(f"radius {r:3d}: CIRCLE coarse centre {zc[i,14,14]:7.0f}  coarse max over map {zc[i].max():7.0f}  fine family max {np.maximum(zf[i],0).max():7.0f} | "
          f"ARC coarse max over map (mean over 12 orientations) {ac[i].reshape(12,-1).max(1).mean():7.0f}  fine family max {af[i].reshape(12,-1).max(1).mean():7.0f}")
i = R.index(20); yy, xx = np.unravel_index(zc[i].argmax(), (28, 28)); print("circle r=20: coarse peak at", (yy, xx))
np.set_printoptions(linewidth=200); print("coarse map, circle r=20 (rows 6-22, cols 6-22, /100):\n", (zc[i][6:23, 6:23] / 100).round().astype(int))
