"""N10 step 2e: split n2's and n3's B-order worth into read bias b1[j], read row W1[j], write column W2[:, j] (choose set)."""
import sys, os, json, copy
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcommon import *
base, mB, mT, Ss, calib = nets(); P = photo_sets(); CH = P["choose"]
pos = [{int(b): k for k, b in enumerate(S.tolist())} for S in Ss]
kB, kT = CH.kl(mB), CH.kl(mT); gap = (kB - kT).mean(); res = {}
for name, g, b in (("n2", 19, 1961), ("n3", 25, 1229)):
    j = pos[g][b]; res[name] = {}
    for part in ("b1", "W1", "W2", "all"):
        m = copy.deepcopy(mB)
        with torch.no_grad():
            if part in ("b1", "all"): L1(m, g).bias[j].copy_(L1(mT, g).bias[j])
            if part in ("W1", "all"): L1(m, g).weight[j].copy_(L1(mT, g).weight[j])
            if part in ("W2", "all"): L2(m, g).weight[:, j].copy_(L2(mT, g).weight[:, j])
        gain = kB - CH.kl(m); res[name][part] = dict(share=gain.mean() / gap, ci=(np.array(boot_ci(gain)) / gap).tolist())
        print(name, part, f"{gain.mean()/gap:.4f}", flush=True)
    res[name]["b1_B_T"] = [float(L1(mB, g).bias[j]), float(L1(mT, g).bias[j])]
    print(name, "b1 B -> T", res[name]["b1_B_T"], flush=True)
json.dump(conv(res), open(os.path.join(OUT, "s2_split.json"), "w"), indent=1)
