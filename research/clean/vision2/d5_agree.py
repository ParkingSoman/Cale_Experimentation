"""D5 signal vs noise in the trained change: cosine between the read (and write) changes of the three runs; share of each
run's change that lies along the others' average; per-neuron agreement."""
import os, numpy as np
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); b17 = m.blocks[17]
U0, W0 = np.array(b17.fc1.weight, np.float64), np.array(b17.fc2.weight, np.float64)
R = {k: np.load(os.path.join(D, f)) for k, f in (("all1e-4", "tr_16_all_0.0001.npz"), ("fc1e-4", "tr_16_fc_0.0001.npz"), ("all3e-5", "tr_16_all_3e-05.npz"))}
dU = {k: v["fc1.weight"].astype(np.float64) - U0 for k, v in R.items()}; dW = {k: v["fc2.weight"].astype(np.float64) - W0 for k, v in R.items()}
cos = lambda a, b: float((a * b).sum() / np.linalg.norm(a) / np.linalg.norm(b)); ks = list(R)
for nm, d in (("read", dU), ("write", dW)):
    print(nm, "change cosines:", {f"{a} vs {b}": round(cos(d[a], d[b]), 3) for i, a in enumerate(ks) for b in ks[i + 1:]})
    pn = [np.array([cos(d[a][j], d[b][j]) if nm == "read" else cos(d[a][:, j], d[b][:, j]) for j in range(3072)]) for a, b in (("all1e-4", "fc1e-4"),)]
    print(f"   per-neuron cos (all1e-4 vs fc1e-4): median {np.median(pn[0]):.3f}, p90 {np.percentile(pn[0], 90):.3f}")
print("bias change cos (all1e-4 vs fc1e-4): %.3f" % cos(R["all1e-4"]["fc1.bias"] - np.array(b17.fc1.bias), R["fc1e-4"]["fc1.bias"] - np.array(b17.fc1.bias)))
for k in ("all1e-4", "all3e-5"):
    for p in ("gamma", "norm.weight", "norm.bias", "conv_dw.weight"):
        o = np.array(dict(__import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(b17.parameters()))[p]); print(f"   {k} {p}: |change|/|orig| {np.linalg.norm(R[k][p] - o) / np.linalg.norm(o):.4f}")
