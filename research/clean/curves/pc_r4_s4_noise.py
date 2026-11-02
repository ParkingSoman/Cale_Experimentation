# serves: DQ19
"""Round 4, S4 check: how much of the table's off-diagonal rms is photo noise? This tests my claim in LADDER 51 that the tail is structure.

Estimator, fixed before the run. Class-row table from each of two disjoint halves of each class's photos (10 and 10), float64.
  noise: rms of log sbar^1 - log sbar^2, off the diagonal, weights (sbar^1 + sbar^2) / 2, with a row constant removed in that weight.
         Each half table has noise; the full table's own noise against a table of many photos is about noise / 2, if the two errors are independent.
  held-out rank: fit a rank-k table on half 1 only, by the KL fit of pc_real_cert.py (warm from the cached fit on all photos would leak, so
         a random start), and score its rms against half 2 and against half 1, at k = 14 and 38. A gap between the two scores is what the fit took from noise.

Reading, fixed before the run. If the noise is near the rank-128 rms (.2), my claim stands. If the noise is above .5, then rank 128's small number is partly a fit
to the noise of these 20 photos, and my claim is withdrawn as stated. Either way the trunk's object is the table on P, so noise is part of the least on P, but it is
not structure a claim could name.

PREDICTION, written before the first run (2026-11-02): noise rms about .9, and held-out rms at k = 38 well above the in-sample one.

Usage: pc_r4_s4_noise.py [--smoke]
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
ITERS = 30 if a.smoke else 800


def fit(ps, fn, it):
    o = torch.optim.LBFGS(ps, lr=1, max_iter=it, history_size=30, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-13)

    def cl():
        o.zero_grad()
        v = fn()
        v.backward()
        return v
    o.step(cl)
    return float(fn().detach())


def rms_off(T, model, weight, off):
    r = T - model
    w = weight * off
    r = r - (w * r).sum(1, keepdim=True) / w.sum(1, keepdim=True)
    return float(((w * r ** 2).sum() / w.sum()).sqrt())


out = {}
for net in ("resnet18", "resnet50"):
    Y = np.load(os.path.join(HERE, f"cache/map/{net}_y.npy"), mmap_mode="r")
    lab = torch.from_numpy(np.load(os.path.join(HERE, f"cache/map/{net}_meta.npz"))["labels"].astype(np.int64))
    S = torch.softmax(torch.from_numpy(np.asarray(Y).astype(np.float64)), 1)
    ids = torch.unique(lab)
    r1, r2 = [], []
    for c in ids:
        ix = (lab == c).nonzero().squeeze(1)
        h = len(ix) // 2
        r1.append(S[ix[:h]].mean(0))
        r2.append(S[ix[h:]].mean(0))
    s1 = torch.stack(r1).clamp_min(1e-12)
    s1 = s1 / s1.sum(1, keepdim=True)
    s2 = torch.stack(r2).clamp_min(1e-12)
    s2 = s2 / s2.sum(1, keepdim=True)
    K, C = s1.shape
    off = torch.ones_like(s1, dtype=torch.bool)
    off[torch.arange(K), ids] = False
    T1, T2 = s1.log(), s2.log()
    row = dict(noise=rms_off(T1, T2, (s1 + s2) / 2, off), in_half1=None)
    print(net, row, flush=True)
    for k in ((14,) if a.smoke else (14, 38)):
        torch.manual_seed(k)
        V = (torch.randn(C, k) * .3).requires_grad_(True)
        c = (torch.randn(K, k) * .3).requires_grad_(True)
        t = T1.mean(0).clone().requires_grad_(True)
        kl = fit([V, c, t], lambda: (s1 * (T1 - torch.log_softmax(t + c @ V.T, 1))).sum(1).mean(), ITERS)
        with torch.no_grad():
            m = t + c @ V.T
            lq = torch.log_softmax(m, 1)
            row[f"k{k}"] = dict(kl_in=kl, kl_out=float((s2 * (T2 - lq)).sum(1).mean()), rms_in=rms_off(T1, m, s1, off), rms_out=rms_off(T2, m, s2, off))
        print(net, k, row[f"k{k}"], flush=True)
    out[net] = row
if not a.smoke:
    with open(os.path.join(HERE, "rounds/round4/s4_noise.json"), "w") as f:
        json.dump(out, f, indent=1)
