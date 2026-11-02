# serves: TQ14
"""Part C. Theorem TQ14 part 6 needs K <= |S| + 1 for a read to write any table of class codes. With K = 1000 class rows, ResNet-18 (512 channels) can never meet it. ResNet-50 (2048) can.
So: is the gap between one read and free codes smaller on ResNet-50? Same photos for both nets: cache/map (20 validation crops per class, not N's training photos).
Estimator, fixed before the run: photos split by a fixed seed, 15000 to fit, 5000 held out. k = 14, 38, 64, 128. Read: logits t + V M z, z the mean pre-activation, standardised, on the fit photos; L-BFGS, 400 iterations, from the logit PCA span with its least-squares read, and also from a random start, keep the lower value on the fit photos. Free codes: k numbers free on each fit photo, 400 iterations, from a random start and from the read's solution, keep the lower. Reported: N's information on the photos, read KL on fit and held-out photos, free KL, the ratio read / free, and the coarse-pole floor (information - log 2k).
The read here is the always-on read, so it is an upper value of V_b(k), not V_b(k) itself.
Prediction, written before the run: the ratio read / free on the fit photos is smaller on ResNet-50 than on ResNet-18 at k = 14 and 38, but stays above 2 at both, because z is not collapsed on either net. On held-out photos the read of ResNet-50 is further from its fit value than on ResNet-18, since it has four times the read weights on the same photos.
Usage: pc_real_wide.py [--smoke] [--threads N]"""
import argparse
import json
import os
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(a.threads)
torch.set_default_dtype(torch.float32)
ITERS = 20 if a.smoke else 400
KS = (14,) if a.smoke else (14, 38, 64, 128)


def fit(params, fn, iters):
    opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, history_size=30, line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-11)

    def cl():
        opt.zero_grad()
        v = fn()
        v.backward()
        for p in params:
            p.grad = p.grad.contiguous()
        return v
    opt.step(cl)
    return float(fn().detach())


out = dict(rows=[], nets={})
for net in ("resnet18", "resnet50"):
    Z = np.load(os.path.join(HERE, f"cache/map/{net}_z.npy"), mmap_mode="r")
    Y = np.load(os.path.join(HERE, f"cache/map/{net}_y.npy"), mmap_mode="r")
    n = Z.shape[0]
    perm = np.random.default_rng(50).permutation(n)
    if a.smoke:
        perm = perm[:2000]
    cut = int(len(perm) * .75)
    tr, te = np.sort(perm[:cut]), np.sort(perm[cut:])
    Zt, Yt = torch.from_numpy(np.asarray(Z[tr])), torch.from_numpy(np.asarray(Y[tr]))
    Ze, Ye = torch.from_numpy(np.asarray(Z[te])), torch.from_numpy(np.asarray(Y[te]))
    mu, sd = Zt.mean(0), Zt.std(0) + 1e-4
    Zt, Ze = (Zt - mu) / sd, (Ze - mu) / sd
    lst, lse = torch.log_softmax(Yt, 1), torch.log_softmax(Ye, 1)
    St, Se = lst.exp(), lse.exp()
    info = float((St * (lst - St.mean(0).log())).sum(1).mean())
    ent = float(-(St * lst).sum(1).mean())
    out["nets"][net] = dict(info=info, entropy=ent, d=Zt.shape[1], top1_conf=float(St.max(1).values.mean()))
    print(net, out["nets"][net], flush=True)

    def kl(L, S=St, ls=lst):
        return (S * (ls - torch.log_softmax(L, 1))).sum(1).mean()

    D = Yt - Yt.mean(1, keepdim=True)
    ybar = D.mean(0)
    D = D - ybar
    ev, U = torch.linalg.eigh((D.T @ D / len(D)).double())
    U = U.flip(1).float().contiguous()
    for k in KS:
        t0 = time.time()
        Q = U[:, :k].contiguous()
        M0 = torch.linalg.lstsq(Zt.double(), (D @ Q).double()).solution.float().contiguous()
        best = None
        for name in ("pca", "random"):
            torch.manual_seed(k)
            V = (Q.clone() if name == "pca" else torch.randn(1000, k) * .1).requires_grad_(True)
            M = (M0.clone() if name == "pca" else torch.randn(Zt.shape[1], k) * .01).requires_grad_(True)
            t = ybar.clone().requires_grad_(True)
            v = fit([V, M, t], lambda: kl(t + (Zt @ M) @ V.T), ITERS)
            print(net, k, "read", name, round(v, 4), round(time.time() - t0), flush=True)
            if best is None or v < best[0]:
                best = (v, V.detach().clone(), M.detach().clone(), t.detach().clone())
        vr, V, M, t = best
        held = float(kl(t + (Ze @ M) @ V.T, Se, lse))
        agree = float(((t + (Zt @ M) @ V.T).argmax(1) == Yt.argmax(1)).float().mean())
        fbest = 1e9
        for name in ("read", "random"):
            torch.manual_seed(k + 1)
            V2 = (V.clone() if name == "read" else torch.randn(1000, k) * .3).requires_grad_(True)
            c2 = ((Zt @ M).clone() if name == "read" else torch.randn(len(Zt), k) * .3).requires_grad_(True)
            t2 = t.clone().requires_grad_(True)
            v = fit([V2, c2, t2], lambda: kl(t2 + c2 @ V2.T), ITERS)
            print(net, k, "free", name, round(v, 4), round(time.time() - t0), flush=True)
            fbest = min(fbest, v)
        row = dict(net=net, k=k, read=vr, read_held_out=held, top1_read=agree, free=fbest, ratio=vr / fbest, coarse_pole_floor=info - float(np.log(2 * k)), sec=time.time() - t0)
        out["rows"].append(row)
        print(row, flush=True)
        if not a.smoke:
            json.dump(out, open(os.path.join(HERE, "results_pc_real_wide.json"), "w"), indent=1)
print("done", flush=True)
