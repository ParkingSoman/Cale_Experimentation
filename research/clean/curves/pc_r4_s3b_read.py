# serves: TQ29
"""Round 4, screen S3b (card 3-37): the same fixed-metric question as S3, on the real read, since P4 reads the box.

Estimator, fixed before the run. cache/map z and y, float64, the rows and standardisation of pc_real_cert.py. Logits t + (z @ M) @ V^T,
the cached read fit at k = 14, 38.
  R(k)  = KL / ((1/2) sum p Var_s(xi)) at the fit.
  Q(k)  = the least found of (1/2) sum p Var_s(T - t - (z @ M) @ V^T) over rank k, by LBFGS, 800 iterations. From the fit, and at k = 14
          from a random start. T = log s. The metric is the teacher's own, fixed. No per-row code, one linear read.
  kl_of_Q = the KL of Q's own solution.

Screen line, fixed before the run: the read's Q is useful as a candidate floor if Q(k) >= fit / 2 at both k on both nets, and the warm
and random starts agree within 5%.

PREDICTION, written before the first run (2026-11-02): Q(k) / fit is between .4 and .8, lower than on the table at the same k, since
the read's fit (about 1.1 nats) sits at a larger amplitude than the table's. Warm and random agree within 5%.

Usage: pc_r4_s3b_read.py [--smoke]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)
ITERS = 20 if a.smoke else 800


def fit(ps, fn, it):
    o = torch.optim.LBFGS(ps, lr=1, max_iter=it, history_size=30, line_search_fn="strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-14)

    def cl():
        o.zero_grad()
        v = fn()
        v.backward()
        return v
    o.step(cl)
    return float(fn().detach())


out = {}
for net in ("resnet18", "resnet50"):
    Zm = np.load(os.path.join(HERE, f"cache/map/{net}_z.npy"), mmap_mode="r")
    Ym = np.load(os.path.join(HERE, f"cache/map/{net}_y.npy"), mmap_mode="r")
    n = 1500 if a.smoke else Zm.shape[0]
    Z = torch.from_numpy(np.asarray(Zm[:n]).astype(np.float64))
    Y = torch.from_numpy(np.asarray(Ym[:n]).astype(np.float64))
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-4)
    T = torch.log_softmax(Y, 1)
    S = T.exp()

    def qform(t, V, M):
        x = T - t - (Z @ M) @ V.T
        m = (S * x).sum(1, keepdim=True)
        return .5 * (S * (x - m) ** 2).sum(1).mean()

    rows = []
    for k in (14, 38):
        t0 = time.time()
        f = np.load(os.path.join(HERE, f"cache/map/fit_{net}_read_{k}.npz"))
        V, M, t = (torch.from_numpy(f[x].astype(np.float64)) for x in ("V", "M", "t"))
        with torch.no_grad():
            lq = torch.log_softmax(t + (Z @ M) @ V.T, 1)
            xi = T - lq
            klr = (S * xi).sum(1)
            var = (S * (xi - (S * xi).sum(1, keepdim=True)) ** 2).sum(1)
        ratio = klr / (.5 * var)
        t1, V1, M1 = t.clone().requires_grad_(True), V.clone().requires_grad_(True), M.clone().requires_grad_(True)
        q_warm = fit([t1, V1, M1], lambda: qform(t1, V1, M1), ITERS)
        with torch.no_grad():
            kl_q = float((S * (T - torch.log_softmax(t1 + (Z @ M1) @ V1.T, 1))).sum(1).mean())
        row = dict(k=k, n=n, fit=float(klr.mean()), half_var=float(.5 * var.mean()), R=float(klr.mean() / (.5 * var.mean())),
                   q10=float(ratio.quantile(.1)), q50=float(ratio.quantile(.5)), q90=float(ratio.quantile(.9)), Q_warm=q_warm, kl_of_Q_warm=kl_q)
        if k == 14:
            torch.manual_seed(k)
            t2, V2, M2 = T.mean(0).clone().requires_grad_(True), (torch.randn(1000, k) * .1).requires_grad_(True), (torch.randn(Z.shape[1], k) * .05).requires_grad_(True)
            row["Q_rand"] = fit([t2, V2, M2], lambda: qform(t2, V2, M2), ITERS)
        row["sec"] = time.time() - t0
        rows.append(row)
        print(net, row, flush=True)
    out[net] = rows
if not a.smoke:
    with open(os.path.join(HERE, "rounds/round4/s3b_read.json"), "w") as f:
        json.dump(out, f, indent=1)
