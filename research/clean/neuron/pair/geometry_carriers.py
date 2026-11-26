"""geometry: reader's question. Is s's mapped read = p's read + the common read of the Base 'carriers' of s's added
rows (s on, p quiet; reader family.json, 8 carriers per pair, 30 pairs)? u_B covariance metric, stream_V reads.
Rules: p alone; NNLS on [p, mean of carriers]; NNLS on [p, 8 carriers]. Controls: carriers of another pair; random rows.
"""
import json, os
import numpy as np, torch
from scipy.optimize import nnls
from cache import OUT
T = torch.float64
d = np.load(os.path.join(OUT, "stream_cov.npz"))
L = torch.linalg.cholesky(torch.tensor(d["Cuu"], dtype=T) + 1e-6 * torch.eye(1024, dtype=T))
Wm = (torch.tensor(d["W1"], dtype=T) @ L).numpy(); Wm /= np.linalg.norm(Wm, axis=1, keepdims=True)
Vm = (torch.tensor(np.load(os.path.join(OUT, "stream_V.npy")), dtype=T) @ L).numpy(); Vm /= np.linalg.norm(Vm, axis=1, keepdims=True)
F = json.load(open(os.path.join(OUT, "reader", "family.json")))["pairs"]
rng = np.random.default_rng(0)
def share(cols, v): return 1 - nnls(np.stack(cols, 1), v)[1] ** 2
R = {k: [] for k in ("p", "p+mean", "p+mean_other", "p+mean_rand", "p+8", "p+8_other", "p+8_rand", "cos_mean_res", "cos_mean_res_other")}
for i, f in enumerate(F):
    s, p, car = f["s"], f["p"], f["carriers"]
    oth = F[(i + 7) % len(F)]["carriers"]; rnd = rng.integers(0, 4096, 8).tolist()
    v = Vm[s]; wp = Wm[p]
    R["p"].append(max(0, wp @ v) ** 2)
    for tag, c in (("", car), ("_other", oth), ("_rand", rnd)):
        m = Wm[c].mean(0)
        R["p+mean" + tag].append(share([wp, m], v))
        R["p+8" + tag].append(share([wp] + [Wm[j] for j in c], v))
    res = v - (wp @ v) * wp
    for tag, c in (("", car), ("_other", oth)):
        m = Wm[c].mean(0); m = m - (wp @ m) * wp
        R["cos_mean_res" + tag].append(float(res @ m / np.linalg.norm(res) / np.linalg.norm(m)))
for k, x in R.items(): print(f"{k:20s} median {np.median(x):.3f}  mean {np.mean(x):.3f}")
