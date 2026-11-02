"""Card W2, paired dev check on all 5000 dev images: Fisher write vs vmr vs Bayes (derived read and trained read)."""
import json, numpy as np, torch
from inception import *
from family import Data
from d4_write import stats, writes
from w_fisher import load_build, real_kl
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"])
fw = torch.load("cache/fisher_writes.pt"); builds = {}
for nm, tag in [("der", "derived2-r1-ks7"), ("trn", "L-k1-r-lr3e-2")]:
    fm = load_build(m, fam, tag); res, _ = writes(m, fam, stats(m, fm, fam, D))
    for w in ["bayes", "vmr"]:
        builds[f"{nm}+{w}"] = (fm, [c.to(DEV) for c in res[w][0]], res[w][1])
    builds[f"{nm}+fisher"] = (fm, [c.to(DEV) for c in fw[f"{nm}+fisher"][0]], [b.to(DEV) for b in fw[f"{nm}+fisher"][1]])
    if nm == "trn":
        tw = fm.weights()[2]; builds["trn+trained"] = (fm, tw, [torch.zeros(c.shape[0], device=DEV) for c in tw])
K = real_kl(m, fam, D, D.dv, builds)
for k, v in K.items(): print(f"{k:12s} dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})")
for a, b in [("der+fisher", "der+vmr"), ("der+fisher", "der+bayes"), ("trn+fisher", "trn+vmr"), ("trn+fisher", "trn+trained")]:
    d = K[a] - K[b]; print(f"paired {a} - {b}: {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})")
np.savez("cache/w2_dev5000.npz", **K)
json.dump({k: float(v.mean()) for k, v in K.items()}, open("../../../results/curves-w2-dev5000.json", "w"), indent=1)
