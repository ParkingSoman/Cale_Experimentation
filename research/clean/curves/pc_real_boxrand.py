# serves: TQ11
"""Part C, V_b(k) on ResNet-18 from real starts. cache/r18: 35000 validation crops, z at 49 positions. Earlier values (1.23, 0.34) came only from warm starts near the always-on read, and rule 5b of the method says a low value from a warm start alone is not a reported value.
Estimator, fixed before the run: k = 14. A unit is relu(m . z(x, s) + beta) at each position, then the mean over positions; logits t + V c(x). Two random starts: random m, V; beta at the median of m . z, so each unit is on at about half of the positions. Adam on batches of 1024 photos, lr 3e-3, 4000 steps, then full-batch L-BFGS, 150 iterations, float32. Reported: KL on the fit photos, on the held-out cached photos, gates on, top-1 agreement with N. Weights saved to cache/r18/box_rand_k14.npz.
Then the form of the best fit, on V after a varimax rotation of its columns: the share of each unit's squared weight on its top 2, top 10 and top 50 classes; the number of classes on each pole holding half of its weight; the principal angle cosines with the logit PCA span. A pole build has share 1 on the top 2.
Prediction, written before the run: the value ends between 1.0 and 1.3, not far below the warm start, with gates on near 0.7, and the share on the top 2 stays below 0.1, since the fit holds more nats than any coarse-pole build can.
Usage: pc_real_boxrand.py [--smoke] [--threads N]"""
import argparse
import json
import os
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--threads", type=int, default=6)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(a.threads)
Zn = np.load(os.path.join(HERE, "cache/r18/z_train.npy"), mmap_mode="r")
Yn = np.load(os.path.join(HERE, "cache/r18/y_train.npy"), mmap_mode="r")
Zs = np.load(os.path.join(HERE, "cache/r18/z_sealed.npy"), mmap_mode="r")
Ys = np.load(os.path.join(HERE, "cache/r18/y_sealed.npy"), mmap_mode="r")
n = 1500 if a.smoke else Zn.shape[0]
Z = torch.from_numpy(np.asarray(Zn[:n]).astype(np.float32)).permute(0, 2, 1).contiguous()
Y = torch.from_numpy(np.asarray(Yn[:n]).astype(np.float32))
ns = 500 if a.smoke else min(Zs.shape[0], 10000)
Z2 = torch.from_numpy(np.asarray(Zs[:ns]).astype(np.float32)).permute(0, 2, 1).contiguous()
Y2 = torch.from_numpy(np.asarray(Ys[:ns]).astype(np.float32))
mu, sd = Z.mean((0, 1)), Z.std((0, 1)) + 1e-4
Z, Z2 = (Z - mu) / sd, (Z2 - mu) / sd
ls, ls2 = torch.log_softmax(Y, 1), torch.log_softmax(Y2, 1)
S, S2 = ls.exp(), ls2.exp()
K = 14
STEPS, LB = (100, 10) if a.smoke else (4000, 150)


def code(z, M, b):
    return torch.relu(z @ M + b).mean(1)


def kl(L, S_, ls_):
    return (S_ * (ls_ - torch.log_softmax(L, 1))).sum(1).mean()


def varimax(A, iters=100):
    p, k = A.shape
    R = np.eye(k)
    d = 0
    for _ in range(iters):
        L = A @ R
        U, s, Vt = np.linalg.svd(A.T @ (L ** 3 - L @ np.diag((L ** 2).sum(0)) / p))
        R = U @ Vt
        d2 = s.sum()
        if d2 < d * (1 + 1e-7):
            break
        d = d2
    return A @ R


out = dict(n=n, rows=[])
best = None
for seed in range(1 if a.smoke else 2):
    torch.manual_seed(seed + 100)
    M = torch.randn(512, K) / np.sqrt(512) * 2
    M = M.clone().requires_grad_(True)
    with torch.no_grad():
        pre = Z[:: max(1, n // 2000)] @ M
        b0 = -pre.reshape(-1, K).median(0).values
    b = b0.clone().requires_grad_(True)
    V = (torch.randn(1000, K) * .3).requires_grad_(True)
    t = (Y - Y.mean(1, keepdim=True)).mean(0).clone().requires_grad_(True)
    t0 = time.time()
    opt = torch.optim.Adam([M, b, V, t], lr=3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    rng = np.random.default_rng(seed)
    for step in range(STEPS):
        ix = torch.from_numpy(rng.integers(0, n, 1024))
        loss = kl(t + code(Z[ix], M, b) @ V.T, S[ix], ls[ix])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
        if step % 1000 == 0:
            print(seed, step, float(loss), round(time.time() - t0), flush=True)
    va = float(kl(t + code(Z, M, b) @ V.T, S, ls).detach())
    print(seed, "after Adam", va, round(time.time() - t0), flush=True)
    lb = torch.optim.LBFGS([M, b, V, t], lr=1, max_iter=LB, history_size=20, line_search_fn="strong_wolfe", tolerance_grad=1e-6, tolerance_change=1e-9)

    def cl():
        lb.zero_grad()
        v = kl(t + code(Z, M, b) @ V.T, S, ls)
        v.backward()
        for p in (M, b, V, t):
            p.grad = p.grad.contiguous()
        return v
    lb.step(cl)
    with torch.no_grad():
        Lt = t + code(Z, M, b) @ V.T
        val = float(kl(Lt, S, ls))
        held = float(kl(t + code(Z2, M, b) @ V.T, S2, ls2))
        on = float((Z[:: max(1, n // 2000)] @ M + b > 0).float().mean())
        agree = float((Lt.argmax(1) == Y.argmax(1)).float().mean())
    row = dict(seed=seed, adam=va, kl=val, held_out=held, on=on, top1=agree, sec=time.time() - t0)
    out["rows"].append(row)
    print(row, flush=True)
    if best is None or val < best[0]:
        best = (val, V.detach().numpy().copy(), M.detach().numpy().copy(), b.detach().numpy().copy(), t.detach().numpy().copy())

val, V, M, b, t = best
A = varimax(V - V.mean(0, keepdims=True))
sh = np.sort(A ** 2, axis=0)[::-1] / (A ** 2).sum(0)
half = []
for j in range(K):
    for sign in (1, -1):
        w = np.sort((np.maximum(sign * A[:, j], 0)) ** 2)[::-1]
        half.append(int(np.searchsorted(np.cumsum(w) / w.sum(), .5) + 1))
D = Y.numpy() - Y.numpy().mean(1, keepdims=True)
D = D - D.mean(0)
U = np.linalg.eigh(D.T @ D / n)[1][:, ::-1][:, :K]
Q = np.linalg.qr(V - V.mean(0, keepdims=True))[0]
cos = np.linalg.svd(Q.T @ U, compute_uv=False)
out["form"] = dict(top2_share_median=float(np.median(sh[:2].sum(0))), top10_share_median=float(np.median(sh[:10].sum(0))), top50_share_median=float(np.median(sh[:50].sum(0))), classes_per_pole_half_median=float(np.median(half)), cos_with_logit_pca=[float(x) for x in cos])
print(out["form"], flush=True)
if not a.smoke:
    np.savez(os.path.join(HERE, "cache/r18/box_rand_k14.npz"), V=V, M=M, b=b, t=t, mu=mu.numpy(), sd=sd.numpy())
    json.dump(out, open(os.path.join(HERE, "results_pc_real_boxrand.json"), "w"), indent=1)
print("done", flush=True)
