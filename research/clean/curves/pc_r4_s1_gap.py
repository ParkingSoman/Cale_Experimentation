# serves: DQ22
"""Round 4, screen S1 (card 3-37): is the weight pattern of a real table near one product in operator norm?

Round 3 asked for a product below every entry of a block, and that form covers about 1/280 of the mass. Definition DQ22 asks in
operator norm: eps_gap = sigma_2 of the doubly scaled table outside each row's near classes.

Estimator, fixed before the run. Class-row tables of ResNet-18 and ResNet-50, as in pc_real_cert.py, float64. Near classes of a
row: its top t classes by mass, t in 0, 1, 5, 10, 20, 50, 100, 200 (they are N's own, with no clustering). Share: the share of the
cached fit's loss on the kept entries, at k = 14 and 38, with entry loss s (e^e - 1 - e), e = log(q / s), which sums to KL.

Screen line, fixed before the run: PASS if some t has sigma_2 <= .5 and share >= .7, on both nets, at k = 14.

PREDICTION, written before the first run (2026-11-02): FAIL. At t = 20, sigma_2 is between .6 and .95, since strangers still
carry family structure. The share at t = 20 is above .7 (LADDER 48 puts .70 of the loss on strangers).

Usage: pc_r4_s1_gap.py [--smoke]
"""
import argparse
import json
import os

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)
out = {}
for net in ("resnet18", "resnet50"):
    Y = np.load(os.path.join(HERE, f"cache/map/{net}_y.npy"), mmap_mode="r")
    lab = np.load(os.path.join(HERE, f"cache/map/{net}_meta.npz"))["labels"]
    n = 1500 if a.smoke else Y.shape[0]
    Yt = torch.from_numpy(np.asarray(Y[:n]).astype(np.float64))
    lab = torch.from_numpy(lab[:n].astype(np.int64))
    S = torch.softmax(Yt, 1)
    ids = torch.unique(lab)
    sb = torch.stack([S[lab == c].mean(0) for c in ids]).clamp_min(1e-12)
    sb = sb / sb.sum(1, keepdim=True)
    K, C = sb.shape
    rows = []
    ents = {}
    for k in (14, 38):
        f = np.load(os.path.join(HERE, f"cache/map/fit_{net}_table_{k}.npz"))
        V, c, t = (torch.from_numpy(f[x].astype(np.float64)) for x in ("V", "c", "t"))
        if c.shape[0] != K:
            ents = None
            break
        e = (torch.softmax(t + c @ V.T, 1) / sb).log()
        ents[k] = sb * (e.exp() - 1 - e) / K
    order = sb.argsort(1, descending=True)
    for tt in (0, 1, 5, 10, 20, 50, 100, 200):
        mask = torch.ones(K, C, dtype=torch.bool)
        if tt:
            mask.scatter_(1, order[:, :tt], False)
        W = sb * mask / K
        ra, rb = W.sum(1), W.sum(0)
        keep = rb > 0
        Wt = W[:, keep] / ra[:, None].sqrt() / rb[keep][None, :].sqrt()
        sv = torch.linalg.svdvals(Wt)
        row = dict(t=tt, mass=float(W.sum()), s1=float(sv[0]), s2=float(sv[1]), s3=float(sv[2]), s5=float(sv[4]), s15=float(sv[14]), s39=float(sv[38]),
                   energy_top15=float((sv[1:16] ** 2).sum() / (sv[1:] ** 2).sum()))
        if ents:
            for k in ents:
                row[f"share_{k}"] = float((ents[k] * mask).sum() / ents[k].sum())
                row[f"kl_{k}"] = float(ents[k].sum())
        rows.append(row)
        print(net, row, flush=True)
    out[net] = rows
if not a.smoke:
    with open(os.path.join(HERE, "rounds/round4/s1_gap.json"), "w") as f:
        json.dump(out, f, indent=1)
