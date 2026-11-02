# serves: TQ11
"""Part C, V_b(k) on ResNet-18's last block: the trunk's own box, with units that can switch. cache/r18/z_train: pre-activations at 49 positions.
A unit is relu(m . z(x, s) + beta) at each position s, then the mean over positions, as in Definition DY1; logits t + V c(x). Estimator, fixed before the run: k = 14 and 38; L-BFGS, 200 iterations, float32; two starts:
  (a) on: from the L_on fit on pooled z with beta large (all units on), the always-on read;
  (b) switch: the same m, V from that fit, with beta chosen so each unit is on at about half of the positions, and output bias refit.
Also sealed photos. Compare with L_on (read of z) and with the free-code floor. Prediction, written before the run: switching lowers the value below L_on, but stays far above the floor (0.18 at k = 14), because a unit's gate is one half-space of z.
Usage: pc_real_box.py [--smoke]"""
import argparse
import json
import os
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--iters", type=int, default=200)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(8)
torch.manual_seed(9)
Zn = np.load(os.path.join(HERE, "cache/r18/z_train.npy"), mmap_mode="r")
Yn = np.load(os.path.join(HERE, "cache/r18/y_train.npy"), mmap_mode="r")
Zs = np.load(os.path.join(HERE, "cache/r18/z_sealed.npy"), mmap_mode="r")
Ys = np.load(os.path.join(HERE, "cache/r18/y_sealed.npy"), mmap_mode="r")
n = 1000 if a.smoke else Zn.shape[0]
Z = torch.from_numpy(np.asarray(Zn[:n]).astype(np.float32)).permute(0, 2, 1).contiguous()   # n x 49 x 512
Y = torch.from_numpy(np.asarray(Yn[:n]).astype(np.float32))
ns = 500 if a.smoke else Zs.shape[0]
Z2 = torch.from_numpy(np.asarray(Zs[:ns]).astype(np.float32)).permute(0, 2, 1).contiguous()
Y2 = torch.from_numpy(np.asarray(Ys[:ns]).astype(np.float32))
mu = Z.mean((0, 1))
sd = Z.std((0, 1)) + 1e-4
Z = (Z - mu) / sd
Z2 = (Z2 - mu) / sd
Zp = Z.mean(1)
Zp2 = Z2.mean(1)
ls = torch.log_softmax(Y, 1)
S = ls.exp()
ls2 = torch.log_softmax(Y2, 1)
S2 = ls2.exp()


def kl(L, S=S, ls=ls):
    return (S * (ls - torch.log_softmax(L, 1))).sum(1).mean()


def code(Z, M, b):
    return torch.relu(Z @ M + b).mean(1)


def fit(params, fn, iters):
    opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe", tolerance_grad=1e-6, tolerance_change=1e-9)

    def cl():
        opt.zero_grad()
        v = fn()
        v.backward()
        for p in params:
            p.grad = p.grad.contiguous()
        return v
    opt.step(cl)
    return float(fn().detach())


Yc = Y - Y.mean(1, keepdim=True)
ybar = Yc.mean(0)
D = Yc - ybar
ev, U = torch.linalg.eigh((D.T @ D / n).double())
U = U.flip(1).float().contiguous()
out = dict(n=n, iters=a.iters, rows=[])
for k in (14, 38):
    Q = U[:, :k].contiguous()
    M0 = torch.linalg.lstsq(Zp.double(), (D @ Q).double()).solution.float().contiguous()
    V = Q.clone().requires_grad_(True)
    M = M0.clone().requires_grad_(True)
    t = ybar.clone().requires_grad_(True)
    t0 = time.time()
    on = fit([V, M, t], lambda: kl(t + (Zp @ M) @ V.T), max(a.iters, 300))
    row = dict(k=k, start="always-on read", kl=on, sealed=float(kl(t + (Zp2 @ M) @ V.T, S2, ls2)), sec=time.time() - t0)
    print(row, flush=True)
    out["rows"].append(row)
    V0, M1, t0_ = V.detach(), M.detach(), t.detach()
    with torch.no_grad():
        pre = Z[:: max(1, n // 2000)] @ M1
    for name, beta in (("on", -pre.reshape(-1, k).min(0).values + 1.0), ("switch", -pre.reshape(-1, k).median(0).values)):
        t0 = time.time()
        b = beta.clone().requires_grad_(True)
        M2 = M1.clone().requires_grad_(True)
        V2 = V0.clone().requires_grad_(True)
        with torch.no_grad():
            c0 = code(Z, M2, b)
            tt = (Y - Y.mean(1, keepdim=True) - c0 @ V2.T).mean(0)
        tv = tt.clone().requires_grad_(True)
        start = float(kl(tv + code(Z, M2, b) @ V2.T))
        val = fit([M2, b, V2, tv], lambda: kl(tv + code(Z, M2, b) @ V2.T), a.iters)
        with torch.no_grad():
            on_frac = float((Z[:: max(1, n // 2000)] @ M2 + b > 0).float().mean())
            sealed = float(kl(tv + code(Z2, M2, b) @ V2.T, S2, ls2))
            agree = float(((tv + code(Z, M2, b) @ V2.T).argmax(1) == Y.argmax(1)).float().mean())
        row = dict(k=k, start=name, value_at_start=start, kl=val, sealed=sealed, on_fraction=on_frac, top1=agree, sec=time.time() - t0)
        print(row, flush=True)
        out["rows"].append(row)
        if not a.smoke:
            json.dump(out, open(os.path.join(HERE, "results_pc_real_box.json"), "w"), indent=1)
print("done", flush=True)
