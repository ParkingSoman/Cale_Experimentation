# serves: TQ11
"""Round 4, S8b (card 3-37): with any decoder, how low can k linear numbers of z go? This estimates the read's information limit from above, on sealed photos.

S8 held M at the read's fit: a flexible decoder gained nothing on sealed photos (ratio 1.005). Here M is trained jointly with the decoder. A linear map to k numbers, then 2 hidden layers of 1024 ReLU units,
then 1000 logits. Two starts: random, and the read's M (refit by LBFGS, as in pc_r4_s8_decoder.py, 300 iterations, which is enough for a start). Adam, 150 epochs, random 10% validation, early stopping.
It is not a network the trunk allows, since the decoder adds layers. It says how much of the read's sealed loss any decoder from k linear numbers could remove. It is an upper value of that least, so it is no floor.
Also the plain read's sealed loss at the same k, for the ratio.

Line, fixed before the run: if the joint model's sealed KL >= .7 x the read's at k = 14, then k linear numbers of z are the limit, whichever decoder reads them, up to this fit's quality.
PREDICTION, written before the first run (2026-11-02): at k = 14 the joint model's sealed KL is 1.25 to 1.45, against the read's 1.49. So the line holds, with ratio above .8.

Usage: pc_r4_s8b_joint.py [--smoke] [--k 14]
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
    return torch.from_numpy(np.asarray(Z[:n], dtype=np.float32).mean(axis=2).astype(np.float64)), torch.from_numpy(np.asarray(Y[:n]).astype(np.float64))


Z, Y = load("train", 1500 if a.smoke else None)
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
o = torch.optim.LBFGS([M, V, t], lr=1, max_iter=30 if a.smoke else 300, history_size=30, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-13)


def cl():
    o.zero_grad()
    v = kl(t + (Z @ M) @ V.T, S, ls)
    v.backward()
    return v


o.step(cl)
with torch.no_grad():
    results = dict(k=k, n=Z.shape[0], read_train_300=float(kl(t + (Z @ M) @ V.T, S, ls)), read_sealed_300=float(kl(t + (Z2 @ M) @ V.T, S2, ls2)))
print(results, flush=True)
M0 = M.detach().float()
torch.set_default_dtype(torch.float32)
Xf, X2f = Z.float(), Z2.float()
Sf, lsf, S2f, ls2f = S.float(), ls.float(), S2.float(), ls2.float()
shuffle = torch.randperm(Xf.shape[0], generator=torch.Generator().manual_seed(7))
Xf, Sf, lsf = Xf[shuffle], Sf[shuffle], lsf[shuffle]
nv = Xf.shape[0] // 10
N = Xf.shape[0]
epochs = 3 if a.smoke else 150
for label in ("from_read", "random"):
    torch.manual_seed(1)
    enc = torch.nn.Linear(512, k, bias=False)
    if label == "from_read":
        with torch.no_grad():
            scale = (Xf @ M0).std(0)
            enc.weight.copy_((M0 / scale).T)
    dec = torch.nn.Sequential(torch.nn.Linear(k, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 1000))
    net = torch.nn.Sequential(enc, dec)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_sealed, best_train = 1e9, None, None
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(N - nv) + nv
        for b in range(0, N - nv, 512):
            ix = perm[b:b + 512]
            opt.zero_grad()
            (Sf[ix] * (lsf[ix] - torch.log_softmax(net(Xf[ix]), 1))).sum(1).mean().backward()
            opt.step()
        sch.step()
        with torch.no_grad():
            val = float((Sf[:nv] * (lsf[:nv] - torch.log_softmax(net(Xf[:nv]), 1))).sum(1).mean())
            if val < best:
                best = val
                best_sealed = float((S2f * (ls2f - torch.log_softmax(net(X2f), 1))).sum(1).mean())
                best_train = float((Sf[nv:] * (lsf[nv:] - torch.log_softmax(net(Xf[nv:]), 1))).sum(1).mean())
        if ep % 25 == 0:
            print(label, ep, round(val, 4), round(best_sealed, 4), round(time.time() - t0), flush=True)
    results[label] = dict(train=best_train, val=best, sealed=best_sealed)
    print(label, results[label], flush=True)
if not a.smoke:
    with open(os.path.join(HERE, f"rounds/round4/s8b_joint_k{k}.json"), "w") as f:
        json.dump(results, f, indent=1)
