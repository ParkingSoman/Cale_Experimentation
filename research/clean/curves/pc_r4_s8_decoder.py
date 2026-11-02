# serves: TQ11
"""Round 4, S8 (card 3-37): is the 14-unit read limited by information, or by its linear decoder? Sealed photos decide.

For any decoder from y = M z, the loss on a law is at least I(i; x | y). So: fit the read (M, V, t) on cache/r18's train photos. Hold M fixed. Train a flexible decoder from y to
1000 logits on the same photos. Compare both on the sealed photos, which the fits never see.
  If the flexible decoder's sealed KL is near the read's, then these 14 numbers hold no more that any decoder could use: the read is limited by information, at this M.
  If it is far lower, the linear decoder is the limit, and a stack of layers or a better code is what the read lacks.

Estimator, fixed before the run. Mean z over positions, standardised on train, float64 for the read (LBFGS, 1500 iterations, from the PCA span, as in pc_r4_s5_gates.py), and float32 for the
decoder: MLP on y with 2 hidden layers of 1024 ReLU units, Adam, 150 epochs, batch 512, with early stopping on a part of train (10%). Also the same MLP on 64 PCA directions of z, as a reference.
Screen line, fixed before the run: "limited by information" if decoder sealed KL >= .7 x read sealed KL.
The result is at one M, and it is not a floor over all M. It is a label ONE-BUILD measurement, and not a network the trunk allows, since this decoder adds layers.

PREDICTION, written before the first run (2026-11-02): read sealed about 1.2. Decoder sealed between .85 and 1.05. So the read is limited by information, at this M, by the line.

Usage: pc_r4_s8_decoder.py [--smoke]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--k", type=int, default=14)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(6)


def load(name, count=None):
    Z = np.load(os.path.join(HERE, f"cache/r18/z_{name}.npy"), mmap_mode="r")
    Y = np.load(os.path.join(HERE, f"cache/r18/y_{name}.npy"), mmap_mode="r")
    n = Z.shape[0] if count is None else count
    pooled = np.asarray(Z[:n], dtype=np.float32).mean(axis=2)
    return torch.from_numpy(pooled.astype(np.float64)), torch.from_numpy(np.asarray(Y[:n]).astype(np.float64))


n = 1500 if a.smoke else None
Z, Y = load("train", n)
Z2, Y2 = load("sealed", 500 if a.smoke else None)
mu, sd = Z.mean(0), Z.std(0) + 1e-4
Z, Z2 = (Z - mu) / sd, (Z2 - mu) / sd
ls, ls2 = torch.log_softmax(Y, 1), torch.log_softmax(Y2, 1)
S, S2 = ls.exp(), ls2.exp()


def kl(L, S, ls):
    return (S * (ls - torch.log_softmax(L, 1))).sum(1).mean()


k = a.k
D = Y - Y.mean(1, keepdim=True)
ybar = D.mean(0)
D = D - ybar
U = torch.linalg.eigh(D.T @ D / D.shape[0])[1].flip(1)
Q0 = U[:, :k].contiguous()
M = torch.linalg.lstsq(Z, D @ Q0).solution.contiguous().requires_grad_(True)
V = Q0.clone().requires_grad_(True)
t = ybar.clone().requires_grad_(True)
o = torch.optim.LBFGS([M, V, t], lr=1, max_iter=30 if a.smoke else 1500, history_size=30, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-13)


def cl():
    o.zero_grad()
    v = kl(t + (Z @ M) @ V.T, S, ls)
    v.backward()
    return v


t0 = time.time()
o.step(cl)
with torch.no_grad():
    read_train, read_sealed = float(kl(t + (Z @ M) @ V.T, S, ls)), float(kl(t + (Z2 @ M) @ V.T, S2, ls2))
print("read", read_train, read_sealed, "sec", time.time() - t0, flush=True)
results = dict(k=k, n=Z.shape[0], ns=Z2.shape[0], read_train=read_train, read_sealed=read_sealed)
torch.set_default_dtype(torch.float32)


def mlp_run(X, X2, label, epochs):
    d = X.shape[1]
    Xf, X2f = X.float(), X2.float()
    sc = Xf.std(0) + 1e-6
    Xf, X2f = Xf / sc, X2f / sc
    Sf, lsf, S2f, ls2f = S.float(), ls.float(), S2.float(), ls2.float()
    nv = Xf.shape[0] // 10
    # train rows are sorted by class, so the validation part must be a random one
    shuffle = torch.randperm(Xf.shape[0], generator=torch.Generator().manual_seed(7))
    Xf, Sf, lsf = Xf[shuffle], Sf[shuffle], lsf[shuffle]
    net = torch.nn.Sequential(torch.nn.Linear(d, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 1000))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_sealed, best_train = 1e9, None, None
    N = Xf.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(N - nv) + nv
        for b in range(0, N - nv, 512):
            ix = perm[b:b + 512]
            opt.zero_grad()
            loss = (Sf[ix] * (lsf[ix] - torch.log_softmax(net(Xf[ix]), 1))).sum(1).mean()
            loss.backward()
            opt.step()
        sched.step()
        with torch.no_grad():
            val = float((Sf[:nv] * (lsf[:nv] - torch.log_softmax(net(Xf[:nv]), 1))).sum(1).mean())
            if val < best:
                best = val
                best_sealed = float((S2f * (ls2f - torch.log_softmax(net(X2f), 1))).sum(1).mean())
                best_train = float((Sf[nv:] * (lsf[nv:] - torch.log_softmax(net(Xf[nv:]), 1))).sum(1).mean())
        if ep % 25 == 0:
            print(label, "epoch", ep, "val", round(val, 4), "best sealed", round(best_sealed, 4), flush=True)
    results[label] = dict(train=best_train, val=best, sealed=best_sealed)
    print(label, results[label], flush=True)


epochs = 3 if a.smoke else 150
with torch.no_grad():
    y, y2 = Z @ M, Z2 @ M
mlp_run(y, y2, "mlp_on_read_code", epochs)
Uz = torch.linalg.eigh(Z.T @ Z / Z.shape[0])[1].flip(1)[:, :64]
mlp_run(Z @ Uz, Z2 @ Uz, "mlp_on_64_pca", epochs)
results["ratio"] = results["mlp_on_read_code"]["sealed"] / read_sealed
print("ratio", results["ratio"], flush=True)
if not a.smoke:
    with open(os.path.join(HERE, f"rounds/round4/s8_decoder_k{k}.json"), "w") as f:
        json.dump(results, f, indent=1)
