# serves: DQ19
"""Round 4, before screen S4 (card 3-37): can any tree reach rms 1.0, the screen's line, on real class-row tables?

A tree weight claim has at most K - 1 numbers f_v beyond the scales. A rank-k model has about k (K + C) numbers, and it holds every
tree only at k about K. So this is a rough comparison and not a theorem: how does the off-diagonal rms of a rank-k fit fall with k?

Estimator, fixed before the run. Table and off-diagonal rms as in pc_real_tree.py: r = log s - model, with a row constant removed in
the s-weighted off-diagonal mean, rms = sqrt(sum s r^2 / sum s) off the diagonal. Models: the cached KL fits at k = 14, 38, 64, 128,
which are not fitted to this rms, so these are upper values for the best rank-k rms.

Decision line, fixed before the run: if rank 38 is already at or above 1.0, agents' trees have a bad outlook: report it before spending
tokens on S4. If rank 14 is well below 1.0, S4 goes ahead.

PREDICTION, written before the first run (2026-11-02): rms about 1.2 at k = 14, about .8 at k = 38, about .4 at k = 128.

Usage: pc_r4_s4_ceiling.py
"""
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)
out = {}
for net in ("resnet18", "resnet50"):
    Y = np.load(os.path.join(HERE, f"cache/map/{net}_y.npy"), mmap_mode="r")
    lab = torch.from_numpy(np.load(os.path.join(HERE, f"cache/map/{net}_meta.npz"))["labels"].astype(np.int64))
    S = torch.softmax(torch.from_numpy(np.asarray(Y).astype(np.float64)), 1)
    ids = torch.unique(lab)
    sb = torch.stack([S[lab == c].mean(0) for c in ids]).clamp_min(1e-12)
    sb = sb / sb.sum(1, keepdim=True)
    K = sb.shape[0]
    T = sb.log()
    off = torch.ones_like(sb, dtype=torch.bool)
    off[torch.arange(K), ids] = False
    rows = []
    for k in (0, 14, 38, 64, 128):
        if k:
            f = np.load(os.path.join(HERE, f"cache/map/fit_{net}_table_{k}.npz"))
            V, c, t = (torch.from_numpy(f[x].astype(np.float64)) for x in ("V", "c", "t"))
            model = t + c @ V.T
        else:
            model = T.mean(0, keepdim=True).expand_as(T)
        r = T - model
        w = sb * off
        r = r - (w * r).sum(1, keepdim=True) / w.sum(1, keepdim=True)
        rms = float(((w * r ** 2).sum() / w.sum()).sqrt())
        within3 = float((w * (r.abs() < np.log(3))).sum() / w.sum())
        rows.append(dict(k=k, rms=rms, within3=within3))
        print(net, rows[-1], flush=True)
    out[net] = rows
with open(os.path.join(HERE, "rounds/round4/s4_ceiling.json"), "w") as f:
    json.dump(out, f, indent=1)
