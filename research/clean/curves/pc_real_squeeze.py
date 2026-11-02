# serves: TQ11
"""Part C, the real object of Definition DQ9 on ResNet-18 (cache/r18, 35000 ImageNet validation crops, weight 1/35000 each;
not N's training set). Estimator, fixed before the run. For each k: joint L-BFGS from the logit PCA span, with a cap on
iterations.
 (a) L_on(k): logits t + V M h(x), one read of the 512 channels. Upper bound on the least.
 (b) L_free(k): logits t + V c_x, codes free at each photo. Upper bound on the floor (a local minimum: not certified).
 (c) the same head's KL on the sealed 10000 photos, to see that (a) is not an artefact of P.
 (d) the proxy's rank check of TQ11 part 3: L_on(k + a - 1) for a = 100, k = 14, 38 (run with --k 113,137).
Prediction, written before the run: L_on(38) a little below the PCA value 1.47; L_free(38) below L_on(38); L_on(137) near
the proxy's 0.06 to 0.1.
Usage: pc_real_squeeze.py --smoke | --k 14,38,64,128"""
import argparse
import json
import os
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--k", default="14,38,64,128")
ap.add_argument("--iters", type=int, default=400)
ap.add_argument("--free", type=int, default=1)
ap.add_argument("--feat", default="h")
ap.add_argument("--out", default="results_pc_real_squeeze.json")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(8)
Yn = np.load(os.path.join(HERE, "cache/r18/y_train.npy"), mmap_mode="r")
Hn = np.load(os.path.join(HERE, "cache/r18/h_train.npy"), mmap_mode="r")
Ys = np.load(os.path.join(HERE, "cache/r18/y_sealed.npy"), mmap_mode="r")
Hs = np.load(os.path.join(HERE, "cache/r18/h_sealed.npy"), mmap_mode="r")
if a.feat == "z":  # the trunk's box reads pre-activations: pool them over the positions. A read of h (after the ReLU) adds a layer (referee, 2026-11-01)
    def pooled(name):
        Z = np.load(os.path.join(HERE, f"cache/r18/{name}.npy"), mmap_mode="r")
        return np.concatenate([np.asarray(Z[i:i + 1000]).astype(np.float32).reshape(len(Z[i:i + 1000]), Z.shape[1], -1).mean(-1) for i in range(0, Z.shape[0], 1000)])
    Hn, Hs = pooled("z_train"), pooled("z_sealed")
n = 1000 if a.smoke else Yn.shape[0]
Y = torch.tensor(np.asarray(Yn[:n]), dtype=torch.float64)
H = torch.tensor(np.asarray(Hn[:n]), dtype=torch.float64)
Y2 = torch.tensor(np.asarray(Ys), dtype=torch.float64)
H2 = torch.tensor(np.asarray(Hs), dtype=torch.float64)
hm, hs = H.mean(0), H.std(0) + 1e-6
Hz = (H - hm) / hs
Hz2 = (H2 - hm) / hs
ls = torch.log_softmax(Y, 1)
S = ls.exp()
ls2 = torch.log_softmax(Y2, 1)
S2 = ls2.exp()


def kl(L, S=S, ls=ls):
    return (S * (ls - torch.log_softmax(L, 1))).sum(1).mean()


Yc = Y - Y.mean(1, keepdim=True)
ybar = Yc.mean(0)
D = Yc - ybar
ev, U = torch.linalg.eigh(D.T @ D / n)
U = U.flip(1)


def fit(params, loss_fn, iters):
    opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, history_size=30, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-11)

    def closure():
        opt.zero_grad()
        v = loss_fn()
        v.backward()
        return v
    opt.step(closure)
    return float(loss_fn().detach())


out = {"n": n, "iters": a.iters, "I_whole": float((S * (ls - torch.log(S.mean(0)))).sum(1).mean()), "rows": []}
print("I whole", out["I_whole"], flush=True)
for k in [int(x) for x in a.k.split(",")]:
    t0 = time.time()
    Q = U[:, :k].contiguous()
    pca = float(kl(ybar + D @ Q @ Q.T))
    M0 = torch.linalg.lstsq(Hz, D @ Q).solution  # 512 x k, read of the channels
    V = Q.contiguous().clone().requires_grad_(True)
    M = M0.contiguous().clone().requires_grad_(True)
    t = ybar.contiguous().clone().requires_grad_(True)
    on = fit([V, M, t], lambda: kl(t + (Hz @ M) @ V.T), a.iters)
    sealed = float(kl(t + (Hz2 @ M) @ V.T, S2, ls2))
    L_on = (t + (Hz @ M) @ V.T).detach()
    row = dict(k=k, pca=pca, L_on=on, L_on_sealed=sealed, sec_on=time.time() - t0,
               top1_agree_on=float((L_on.argmax(1) == Y.argmax(1)).float().mean()))
    print(row, flush=True)
    if a.free:
        t0 = time.time()
        V2 = V.detach().clone().requires_grad_(True)
        t2 = t.detach().clone().requires_grad_(True)
        c = (Hz @ M.detach()).clone().requires_grad_(True)
        free = fit([V2, t2, c], lambda: kl(t2 + c @ V2.T), a.iters)
        L_f = (t2 + c @ V2.T).detach()
        row.update(L_free=free, sec_free=time.time() - t0, code_norm=float(c.norm(dim=1).mean()),
                   top1_agree_free=float((L_f.argmax(1) == Y.argmax(1)).float().mean()))
        print(row, flush=True)
    out["rows"].append(row)
    json.dump(out, open(os.path.join(HERE, a.out), "w"), indent=1)
print("done", flush=True)
