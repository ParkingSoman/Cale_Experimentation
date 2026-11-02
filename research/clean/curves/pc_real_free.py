# serves: TQ11
"""Part C, the circuit question on ResNet-18 (cache/r18, 35000 validation crops). pc_real_squeeze.py started L_free from the one-read solution, one start, so its gap is only a lower bound on the true gap. Estimator, fixed before the run:
free codes at k = 14 and 38, 500 iterations, from (a) a random span and random codes, (b) a class table start: V, c from a table fit on N's top-class regions (mean softmax per top class, K = number of top classes seen), each photo starting at its region's code.
Prints L_free from each start, with the eps_const of the top-class regions. Prediction, written before the run: (b) falls well below the old L_free, as on the CIFAR net (a table fit stops much lower than a read)."""
import json
import os
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(4)
torch.manual_seed(5)
torch.set_default_dtype(torch.float64)
Y = torch.tensor(np.asarray(np.load(os.path.join(HERE, "cache/r18/y_train.npy"), mmap_mode="r")), dtype=torch.float64)
n, C = Y.shape
ls = torch.log_softmax(Y, 1)
S = ls.exp()
top = S.argmax(1)
ids, inv = torch.unique(top, return_inverse=True)
K = len(ids)
pi = torch.bincount(inv, minlength=K).double() / n
sb = torch.zeros(K, C).index_add_(0, inv, S) / (pi * n).unsqueeze(1)
sb = sb.clamp_min(1e-12)
sb = sb / sb.sum(1, keepdim=True)
eps = float((S * (ls - sb[inv].log())).sum(1).mean())
print("regions", K, "eps_const", eps, flush=True)


def kl(L):
    return (S * (ls - torch.log_softmax(L, 1))).sum(1).mean()


def fit(params, fn, iters):
    opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, history_size=30, line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-12)

    def cl():
        opt.zero_grad()
        v = fn()
        v.backward()
        for p in params:
            p.grad = p.grad.contiguous()
        return v
    opt.step(cl)
    return float(fn().detach())


out = dict(K=K, eps_const=eps, rows=[])
for k in (14, 38):
    t0 = time.time()
    V = (torch.randn(C, k) * .3).requires_grad_(True)
    c = (torch.randn(n, k) * .3).requires_grad_(True)
    t = Y.mean(0).clone().requires_grad_(True)
    r = fit([V, c, t], lambda: kl(t + c @ V.T), 500)
    row = dict(k=k, start="random", L_free=r, top1=float(((t + c @ V.T).argmax(1) == Y.argmax(1)).double().mean()), sec=time.time() - t0)
    print(row, flush=True)
    out["rows"].append(row)
    t0 = time.time()
    Vt = (torch.randn(C, k) * .3).requires_grad_(True)
    ct = (torch.randn(K, k) * .3).requires_grad_(True)
    tt = Y.mean(0).clone().requires_grad_(True)
    tv = fit([Vt, ct, tt], lambda: (pi * (sb * (sb.log() - torch.log_softmax(tt + ct @ Vt.T, 1))).sum(1)).sum(), 800)
    V2 = Vt.detach().clone().requires_grad_(True)
    t2 = tt.detach().clone().requires_grad_(True)
    c2 = ct.detach()[inv].clone().requires_grad_(True)
    start_v = float(kl(t2 + c2 @ V2.T))
    r2 = fit([V2, c2, t2], lambda: kl(t2 + c2 @ V2.T), 500)
    row = dict(k=k, start="table", L_table=tv, table_plus_eps=tv + eps, value_at_start=start_v, L_free=r2, top1=float(((t2 + c2 @ V2.T).argmax(1) == Y.argmax(1)).double().mean()), code_norm=float(c2.norm(dim=1).mean()), sec=time.time() - t0)
    print(row, flush=True)
    out["rows"].append(row)
    json.dump(out, open(os.path.join(HERE, "results_pc_real_free.json"), "w"), indent=1)
print("done", flush=True)
