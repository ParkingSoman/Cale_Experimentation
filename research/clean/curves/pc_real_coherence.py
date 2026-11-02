# serves: TQ18
"""Part C. Step 1 of the plan: is the real squeeze coherent, and how clean is it in each photo's own metric? cache/map, 20000 crops; cache/r18 for the box.
Estimator, fixed before the run.
Fits: always-on read of mean z from the logit PCA start and from a random start (lower value kept), free codes from a random start, class-row table from a random start; L-BFGS 400 iterations, float64; k = 14, 38, 64, 128; ResNet-18 and ResNet-50. V and the read saved to cache/map/fit_<net>_<object>_<k>.npz. The box at k = 14 is cache/r18/box_rand_k14.npz, run on cache/r18 photos.
Coherence: rotate V by varimax. For each pole take its top 30 classes. Within-pole similarity = mean cosine of N's own class weight vectors (centred). Control, fixed before the run: the same number with the 1000 rows of V shuffled (one shuffle per fit). Also corr(V_i . V_j, class similarity).
Own-metric cleanliness: for each photo, chi = sum over i of (s_i - q_i)^2 / q_i, write = the variance under q of the build's logits; number = sqrt(chi / write). Also |log(s/q)| on the top 5, 15, 100 classes of s. Also the same with chi cut to the top 100 classes of s, since q can be far too low on a class with real mass.
Prediction, written before the run: the shuffled control is within 0.01 of 0 and the real value above 0.05 at every k and object. The number is near 1 at k = 38 and well below it at k = 128 on the read, and fails at k = 14. The box at k = 14 looks like the read at k = 14. Table and free numbers are smaller than the read's at each k.
Usage: pc_real_coherence.py [--smoke]"""
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
torch.set_num_threads(8)
torch.set_default_dtype(torch.float64)
ITERS = 25 if a.smoke else 400
KS = (14,) if a.smoke else (14, 38, 64, 128)


def fit(ps, fn, it):
    o = torch.optim.LBFGS(ps, lr=1, max_iter=it, history_size=30, line_search_fn="strong_wolfe", tolerance_grad=1e-8, tolerance_change=1e-12)

    def cl():
        o.zero_grad()
        v = fn()
        v.backward()
        for p in ps:
            p.grad = p.grad.contiguous()
        return v
    o.step(cl)
    return float(fn().detach())


def varimax(A, it=200):
    p, k = A.shape
    R = np.eye(k)
    o = 0
    for _ in range(it):
        L = A @ R
        U, s, Vt = np.linalg.svd(A.T @ (L ** 3 - L @ np.diag((L ** 2).sum(0)) / p))
        R = U @ Vt
        if s.sum() < o * (1 + 1e-7):
            break
        o = s.sum()
    return A @ R


def coherence(V, Sim, rng, top=30):
    V = V - V.mean(0)
    A = varimax(V)

    def stat(A_):
        w = []
        for j in range(A_.shape[1]):
            o = np.argsort(A_[:, j])
            for g in (o[::-1][:top], o[:top]):
                w.append(Sim[np.ix_(g, g)][np.triu_indices(top, 1)].mean())
        return float(np.mean(w))
    G = V @ V.T
    off = ~np.eye(1000, dtype=bool)
    return dict(within=stat(A), control=stat(A[rng.permutation(1000)]), corr=float(np.corrcoef(G[off], Sim[off])[0, 1]))


def cleanliness(S, ls, L):
    lq = torch.log_softmax(L, 1)
    q = lq.exp()
    kl = float((S * (ls - lq)).sum(1).mean())
    order = torch.argsort(S, 1, descending=True)
    lr = (ls - lq).abs()
    ratio = (S - q) ** 2 / q.clamp_min(1e-300)
    chi = ratio.sum(1)
    chi100 = torch.gather(ratio, 1, order[:, :100]).sum(1)
    write = (q * (L - (q * L).sum(1, keepdim=True)) ** 2).sum(1)
    num = (chi / write.clamp_min(1e-30)).sqrt()
    num100 = (chi100 / write.clamp_min(1e-30)).sqrt()
    out = dict(kl=kl, number_median=float(num.median()), number_p90=float(num.quantile(.9)), number_max=float(num.max()), number100_median=float(num100.median()), number100_p90=float(num100.quantile(.9)), share_below_1=float((num < 1).double().mean()), write_median=float(write.median()))
    for m in (5, 15, 100):
        v = torch.gather(lr, 1, order[:, :m])
        out[f"top{m}_median"], out[f"top{m}_p90"] = float(v.median()), float(v.quantile(.9))
    return out


out = dict(rows=[])
rng = np.random.default_rng(5)
for net in ("resnet18", "resnet50"):
    meta = np.load(os.path.join(HERE, f"cache/map/{net}_meta.npz"))
    W = meta["W"].astype(np.float64)
    C_ = W - W.mean(0)
    C_ = C_ / np.linalg.norm(C_, axis=1, keepdims=True)
    Sim = C_ @ C_.T
    Yn = np.load(os.path.join(HERE, f"cache/map/{net}_y.npy"), mmap_mode="r")
    Zn = np.load(os.path.join(HERE, f"cache/map/{net}_z.npy"), mmap_mode="r")
    n = 1500 if a.smoke else Yn.shape[0]
    Y = torch.from_numpy(np.asarray(Yn[:n]).astype(np.float64))
    Z = torch.from_numpy(np.asarray(Zn[:n]).astype(np.float64))
    lab = torch.from_numpy(meta["labels"][:n].astype(np.int64))
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-4)
    ls = torch.log_softmax(Y, 1)
    S = ls.exp()
    D = Y - Y.mean(1, keepdim=True)
    yb = D.mean(0)
    D = D - yb
    U = torch.linalg.eigh(D.T @ D / n)[1].flip(1)
    ids = torch.unique(lab)
    sb = torch.stack([S[lab == c].mean(0) for c in ids]).clamp_min(1e-12)
    sb = sb / sb.sum(1, keepdim=True)
    lb = sb.log()
    for k in KS:
        t0 = time.time()
        torch.manual_seed(k)
        best = None
        for start in ("pca", "random"):
            V = (U[:, :k].contiguous().clone() if start == "pca" else torch.randn(1000, k) * .1).requires_grad_(True)
            M = (torch.linalg.lstsq(Z, D @ U[:, :k]).solution.contiguous() if start == "pca" else torch.randn(Z.shape[1], k) * .01).requires_grad_(True)
            t = yb.clone().requires_grad_(True)
            v = fit([V, M, t], lambda: (S * (ls - torch.log_softmax(t + (Z @ M) @ V.T, 1))).sum(1).mean(), ITERS)
            if best is None or v < best[0]:
                best = (v, V.detach().clone(), M.detach().clone(), t.detach().clone())
        v, V, M, t = best
        if not a.smoke:
            np.savez(os.path.join(HERE, f"cache/map/fit_{net}_read_{k}.npz"), V=V.numpy(), M=M.numpy(), t=t.numpy())
        row = dict(net=net, object="read", k=k, **cleanliness(S, ls, t + (Z @ M) @ V.T), **coherence(V.numpy(), Sim, rng), sec=time.time() - t0)
        out["rows"].append(row)
        print(row, flush=True)
        t0 = time.time()
        V2 = V.clone().requires_grad_(True) if False else (torch.randn(1000, k) * .3).requires_grad_(True)
        c2 = (torch.randn(n, k) * .3).requires_grad_(True)
        t2 = yb.clone().requires_grad_(True)
        fit([V2, c2, t2], lambda: (S * (ls - torch.log_softmax(t2 + c2 @ V2.T, 1))).sum(1).mean(), ITERS)
        row = dict(net=net, object="free", k=k, **cleanliness(S, ls, (t2 + c2 @ V2.T).detach()), **coherence(V2.detach().numpy(), Sim, rng), sec=time.time() - t0)
        out["rows"].append(row)
        print(row, flush=True)
        t0 = time.time()
        V3 = (torch.randn(1000, k) * .3).requires_grad_(True)
        c3 = (torch.randn(len(ids), k) * .3).requires_grad_(True)
        t3 = lb.mean(0).clone().requires_grad_(True)
        fit([V3, c3, t3], lambda: (sb * (lb - torch.log_softmax(t3 + c3 @ V3.T, 1))).sum(1).mean(), ITERS * 2)
        if not a.smoke:
            np.savez(os.path.join(HERE, f"cache/map/fit_{net}_table_{k}.npz"), V=V3.detach().numpy(), c=c3.detach().numpy(), t=t3.detach().numpy())
        row = dict(net=net, object="table", k=k, **cleanliness(sb, lb, (t3 + c3 @ V3.T).detach()), **coherence(V3.detach().numpy(), Sim, rng), sec=time.time() - t0)
        out["rows"].append(row)
        print(row, flush=True)
        if not a.smoke:
            json.dump(out, open(os.path.join(HERE, "results_pc_real_coherence.json"), "w"), indent=1)
    if net == "resnet18" and not a.smoke:
        # the box at k = 14, on its own photos
        d = np.load(os.path.join(HERE, "cache/r18/box_rand_k14.npz"))
        Zb = np.load(os.path.join(HERE, "cache/r18/z_train.npy"), mmap_mode="r")
        Yb = np.load(os.path.join(HERE, "cache/r18/y_train.npy"), mmap_mode="r")
        idx = np.arange(0, Zb.shape[0], 5)
        Zz = torch.from_numpy(np.asarray(Zb[idx]).astype(np.float32)).permute(0, 2, 1).contiguous()
        Zz = (Zz - torch.tensor(d["mu"])) / torch.tensor(d["sd"])
        code = torch.relu(Zz @ torch.tensor(d["M"]) + torch.tensor(d["b"])).mean(1).double()
        Yy = torch.from_numpy(np.asarray(Yb[idx]).astype(np.float64))
        lsy = torch.log_softmax(Yy, 1)
        row = dict(net=net, object="box", k=14, **cleanliness(lsy.exp(), lsy, torch.tensor(d["t"]).double() + code @ torch.tensor(d["V"]).double().T), **coherence(d["V"].astype(np.float64), Sim, rng))
        out["rows"].append(row)
        print(row, flush=True)
if not a.smoke:
    json.dump(out, open(os.path.join(HERE, "results_pc_real_coherence.json"), "w"), indent=1)
print("done", flush=True)
