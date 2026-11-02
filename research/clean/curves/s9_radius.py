"""Whose radius? Fine family: preferred arc radius (10-100 px, best orientation per neuron, best spot near centre).
Coarse: preferred centred-circle radius. Also the coarse neuron's input window: 5x5 positions of the 5x5 bottleneck."""
import sys, numpy as np, torch
from inception import *
from family import Family
from stimuli import arc, circle
m = load_model(); S1 = np.load("cache/s1_fine.npz"); fam = list(S1["fam"])
R = [8, 12, 16, 20, 25, 30, 40, 50, 60, 75, 90, 110]
th = np.arange(0, 360, 15)
best = np.zeros((len(R), 10)); full = np.zeros((len(R), 10))
with torch.no_grad():
    for i, r in enumerate(R):
        z = F.relu(pre_3b(m, to_3a(m, prep(np.stack([arc(t, radius=r) for t in th])))))[:, fam, 11:18, 11:18].amax((2, 3)).cpu().numpy()
        best[i] = z.max(0)
        zc = F.relu(pre_3b(m, to_3a(m, prep(circle(radius=r)[None]))))[0, fam].amax((1, 2)).cpu().numpy(); full[i] = zc
np.set_printoptions(precision=0, suppress=True, linewidth=170)
print("radius:", R)
print("fine, best arc response per neuron (rows radius):\n", best.T)
print("fine preferred arc radius per neuron:", [R[j] for j in best.argmax(0)])
print("fine family max on a full circle (any position) by radius:", full.max(1))
if len(sys.argv) > 1:
    ck = torch.load(f"cache/ckpt/{sys.argv[1]}.pt", map_location="cpu"); fm = Family(m, ck["fam"], ck["k"], ks=ck.get("ks", 5)).to(DEV); fm.load_state_dict(ck["state"])
    with torch.no_grad():
        cz = [fm.pre(inputs_3b(m, to_3a(m, prep(circle(radius=r)[None]))))[0, 0, 14, 14].item() for r in R]
    print("coarse at centre of centred circle by radius:", np.array(cz), " preferred:", R[int(np.argmax(cz))])
