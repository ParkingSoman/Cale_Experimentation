# serves: TQ29
"""Round 4, screen S3 (card 3-37): is a real least a fixed-metric quadratic problem, at the network's own amplitude?

Lemma TQ29: KL(s || q) = (1/2) Var_s(xi) exactly when the path variance does not change, as for a Gaussian xi. Real tables are
"quadratic on 8% to 26% of the loss" entry by entry (LADDER 50). This asks a different question, row by row.

Estimator, fixed before the run. Class-row tables of ResNet-18 and ResNet-50, float64, with cached fits at k = 14, 38, 64, 128:
  R(k)  = KL / ((1/2) sum p Var_s(xi)) at the fit, and the row quantiles of KL / ((1/2) Var_s(xi)).
  Q(k)  = the least found of (1/2) sum p Var_s(T - t - V c) over rank k, by LBFGS, from the fit and from a random start. T = log s.
          The metric is the teacher's own, fixed. This is weighted fitting, with no softmax.
  P0(k) = (1/2) the sum over j > k of sigma_j^2 of the table log s, double-centred and scaled by row and column mass. Closed form.

Screen line, fixed before the run: PASS if R(k) is within 10% of 1, Q(k) within 10% of the fit, or P0(k) within 10% of the fit, at
every k, on both nets. Each is a separate line.

PREDICTION, written before the first run (2026-11-02): R(k) is between .5 and 1.2, and it fails the 10% line at some k. Q(k) from
the fit is lower than the fit, since a down-move on a light class costs Var much more than KL only in one direction, and this has
no clear sign, so I predict within 30% only. P0 misses by more than 10%, since the closed span already missed by 2.2 times.

Usage: pc_r4_s3_var.py [--smoke]
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
ITERS = 30 if a.smoke else 1500


def fit(ps, fn, it):
    o = torch.optim.LBFGS(ps, lr=1, max_iter=it, history_size=30, line_search_fn="strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-14)

    def cl():
        o.zero_grad()
        v = fn()
        v.backward()
        return v
    o.step(cl)
    return float(fn().detach())


def qform(sb, T, t, V, c, K):
    x = T - t - c @ V.T
    m = (sb * x).sum(1, keepdim=True)
    return .5 * (sb * (x - m) ** 2).sum(1).sum() / K


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
    T = sb.log()
    W = sb / K
    ra, rb = W.sum(1), W.sum(0)
    rowc = (W * T).sum(1, keepdim=True) / ra[:, None]
    colc = (W * (T - rowc)).sum(0, keepdim=True) / rb[None, :]
    E = T - rowc - colc
    sv = torch.linalg.svdvals(E * ra[:, None].sqrt() * rb[None, :].sqrt())
    rows = []
    for k in (14, 38, 64, 128):
        f = np.load(os.path.join(HERE, f"cache/map/fit_{net}_table_{k}.npz"))
        V, c, t = (torch.from_numpy(f[x].astype(np.float64)) for x in ("V", "c", "t"))
        if c.shape[0] != K:
            print("skip: fit is on other rows", net, k)
            continue
        lq = torch.log_softmax(t + c @ V.T, 1)
        xi = T - lq
        klr = (sb * xi).sum(1)
        m = (sb * xi).sum(1, keepdim=True)
        var = (sb * (xi - m) ** 2).sum(1)
        ratio = klr / (.5 * var)
        p0 = float(.5 * (sv[k:] ** 2).sum())
        t1, V1, c1 = t.clone().requires_grad_(True), V.clone().requires_grad_(True), c.clone().requires_grad_(True)
        q_warm = fit([t1, V1, c1], lambda: qform(sb, T, t1, V1, c1, K), ITERS)
        torch.manual_seed(k)
        t2, V2, c2 = T.mean(0).clone().requires_grad_(True), (torch.randn(C, k) * .3).requires_grad_(True), (torch.randn(K, k) * .3).requires_grad_(True)
        q_rand = fit([t2, V2, c2], lambda: qform(sb, T, t2, V2, c2, K), ITERS)
        # the KL of the quadratic problem's own solution: how good a build does the fixed metric give?
        with torch.no_grad():
            lqq = torch.log_softmax(t1 + c1 @ V1.T, 1)
            kl_of_q = float((sb * (T - lqq)).sum(1).mean())
            lqr = torch.log_softmax(t2 + c2 @ V2.T, 1)
            kl_of_qr = float((sb * (T - lqr)).sum(1).mean())
        row = dict(k=k, fit=float(klr.mean()), half_var=float(.5 * var.mean()), R=float(klr.mean() / (.5 * var.mean())),
                   q10=float(ratio.quantile(.1)), q50=float(ratio.quantile(.5)), q90=float(ratio.quantile(.9)),
                   Q_warm=q_warm, Q_rand=q_rand, kl_of_Q_warm=kl_of_q, kl_of_Q_rand=kl_of_qr, P0=p0)
        rows.append(row)
        print(net, row, flush=True)
    out[net] = rows
if not a.smoke:
    with open(os.path.join(HERE, "rounds/round4/s3_var.json"), "w") as f:
        json.dump(out, f, indent=1)
