# serves: TQ18
"""Part C, the trained small net against Theorems TQ17 and TQ18. cache/map/cifar_net.npz: the net of model/ts_identity.py (CIFAR-100, 50000 training photos, so here P is what N was trained on). Rows: one per class, mean softmax on that class. K = C = 100.
Estimator, fixed before the run. Table fits at k = 2, 5, 10, 20, 40, 50 from 8 random starts, L-BFGS, float64, 2000 iterations. Reported at the best start: KL, the spread of the 8 values, stationarity errors, g, sigma, rho, the test ratio g / (rho min(1, sigma/2)). At k = 50: the pole form, with the pairing by greatest confusion (a measure of form, not a map), its fit against the free fit. Also the read of h at the same k, from the table's solution, on all photos: read KL against table KL.
Prediction, written before the run: the test ratio is above 100 at every k, because this net is overfit and so its probabilities reach very small values. The pole span is above the free fit at k = 50. The spread of the 8 values is small at k <= 10.
Usage: tq18_small.py [--smoke]"""
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
torch.set_default_dtype(torch.float64)
torch.set_num_threads(3)
torch.manual_seed(1)
dat = np.load(os.path.join(HERE, "..", "cache", "map", "cifar_net.npz"))
h = torch.tensor(dat["h"]).double()
lab = torch.tensor(dat["lab"]).long()
W = torch.tensor(dat["W"]).double()
b = torch.tensor(dat["b"]).double()
s = torch.softmax(h @ W.T + b, 1)
sb = torch.stack([s[lab == c].mean(0) for c in range(100)])
print("least table entry", float(sb.min()), flush=True)
sb = sb.clamp_min(1e-300)
sb = sb / sb.sum(1, keepdim=True)
lb = sb.log()
pi = torch.ones(100) / 100
ITERS = 100 if a.smoke else 2000
STARTS = 2 if a.smoke else 8


def fit(params, fn):
    opt = torch.optim.LBFGS(params, lr=1, max_iter=ITERS, history_size=40, line_search_fn="strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-14)

    def cl():
        opt.zero_grad()
        v = fn()
        v.backward()
        for p in params:
            p.grad = p.grad.contiguous()
        return v
    opt.step(cl)
    return float(fn().detach())


def tab(t, c, V):
    return (pi * (sb * (lb - torch.log_softmax(t + c @ V.T, 1))).sum(1)).sum()


out = dict(rows=[])
for k in ((2, 5) if a.smoke else (2, 5, 10, 20, 40, 50)):
    t0 = time.time()
    vals, best = [], None
    for j in range(STARTS):
        V = (torch.randn(100, k) * .5).requires_grad_(True)
        c = (torch.randn(100, k) * .5).requires_grad_(True)
        t = lb.mean(0).clone().requires_grad_(True)
        v = fit([V, c, t], lambda: tab(t, c, V))
        vals.append(v)
        if best is None or v < best[0]:
            best = (v, t.detach().clone(), c.detach().clone(), V.detach().clone())
    v, t, c, V = best
    mean = c.mean(0, keepdim=True)
    t2 = t + (mean @ V.T)[0]
    c2 = c - mean
    Vc = V - V.mean(0, keepdim=True)
    Q, Rr = torch.linalg.qr(Vc)
    c2 = c2 @ Rr.T
    q = torch.softmax(t2 + c2 @ Q.T, 1)
    G = pi[:, None] * (sb - q)
    g = float(torch.linalg.matrix_norm(G, 2))
    sigma = float(torch.linalg.svdvals(c2).min())
    rho = float((pi[:, None] * q).min())
    row = dict(k=k, kl=v, spread=max(vals) - min(vals), median_over_least=float(np.median(vals) / v), stat_mean=float(G.sum(0).abs().sum()), stat_c=float((c2.T @ G).norm()), stat_V=float((G @ Q).norm()), g=g, sigma=sigma, rho=rho, ratio=g / (rho * min(1., sigma / 2)), q_min=float(q.min()), sec=time.time() - t0)
    out["rows"].append(row)
    print(row, flush=True)
if not a.smoke:
    conf = sb + sb.T
    conf.fill_diagonal_(-1)
    pair = -np.ones(100, dtype=int)
    for f in torch.argsort(conf.flatten(), descending=True).tolist():
        i, j = f // 100, f % 100
        if pair[i] < 0 and pair[j] < 0 and i != j:
            pair[i], pair[j] = j, i
    Vp = torch.zeros(100, 50)
    j = 0
    for c_ in range(100):
        if c_ < pair[c_]:
            Vp[c_, j], Vp[pair[c_], j] = 1, -1
            j += 1
    cp = torch.zeros(100, 50).requires_grad_(True)
    tp = lb.mean(0).clone().requires_grad_(True)
    out["pole"] = dict(pole_span_fit=fit([cp, tp], lambda: tab(tp, cp, Vp)), free_fit_k50=[r for r in out["rows"] if r["k"] == 50][0]["kl"])
    print(out["pole"], flush=True)
    json.dump(out, open(os.path.join(HERE, "tq18_small.json"), "w"), indent=1)
print("done", flush=True)
