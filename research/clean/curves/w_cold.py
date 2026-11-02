"""ONE sealed test (10000 sealed images) of the write cards: derived read + Bayes / vmr / Fisher (W2, 35000 images)
/ active-set step 1 (W5, 35000 images), against the trained k=1 ruler. Paired differences on the same images."""
import json, numpy as np, torch
from inception import *
from family import Data
from d4_write import stats, writes
from w_fisher import load_build, real_kl
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"])
fd = load_build(m, fam, "derived2-r1-ks7"); ft = load_build(m, fam, "L-k1-r-lr3e-2"); tw = ft.weights()[2]
res, _ = writes(m, fam, stats(m, fd, fam, D)); B = {}
for w in ["bayes", "vmr"]: B[w] = (fd, [c.to(DEV) for c in res[w][0]], res[w][1])
for nm, f in [("fisher", "cache/fisher_write_derived2-r1-ks7_35000.pt"), ("active1", "cache/fisher_active_derived2-r1-ks7_35000_s1.pt")]:
    c, b = torch.load(f); B[nm] = (fd, [x.to(DEV) for x in c], [x.to(DEV) for x in b])
B["trained"] = (ft, tw, [torch.zeros(c.shape[0], device=DEV) for c in tw])
K = real_kl(m, fam, D, D.se, B)
for k, v in K.items(): print(f"{k:8s} sealed KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})", flush=True)
for a, b in [("fisher", "vmr"), ("active1", "fisher"), ("active1", "vmr"), ("active1", "trained")]:
    d = K[a] - K[b]; print(f"paired {a} - {b}: {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})", flush=True)
np.savez("cache/w_cold_per_image.npz", **K)
json.dump({k: float(v.mean()) for k, v in K.items()}, open("../../../results/curves-w-cold.json", "w"), indent=1)
