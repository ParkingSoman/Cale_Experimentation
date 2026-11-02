# serves: TQ22
"""Part C. How much of the real loss do the minors of Proposition TQ22 reach? cache/map, class-row tables of ResNet-18 and ResNet-50.
Estimator, fixed before the run. A partition of the classes into families is needed. This is a measure of form, not a map: agglomerative clusters of N's own class weight vectors (Ward), at J = 2, 5, 10, 20 families, and a random partition of the same sizes as control. For each k in 14, 38: fit the whole table (random start), then each minor: rows = all classes' rows, weights pi s_c(H_b), table s_c | H_b, classes H_b, with k units; L-BFGS 400 iterations, 2 starts, lower kept. Sum of the minors' fits against the whole table's fit. Values are fits, so each minor's value is an upper value of its least: the sum is NOT a proved bound here. It says how much a certified bound could reach at best. Reported also: at each J, the part of the loss between families (the KL of the family masses at the whole fit), and the share inside.
Prediction, written before the run: the sum falls fast as J grows, since small families have rank below k. At J = 2 it reaches over half of the fit at k = 14. A random partition reaches more at J = 2 than a cluster partition, since unrelated classes mix, and less of the between-family part. At J = 20, the minors reach under a tenth at k = 38. Most of the real loss at small J is inside families, not between them.
Usage: pc_real_minors.py [--smoke]"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(6)
torch.set_default_dtype(torch.float64)
ITERS = 30 if a.smoke else 400
rng = np.random.default_rng(9)


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


def table_fit(sb, w, k, starts=1):
    lb = sb.log()
    best = (1e9, None)
    n, C = sb.shape
    for s_ in range(starts):
        V = (torch.randn(C, k) * .3).requires_grad_(True)
        c = (torch.randn(n, k) * .3).requires_grad_(True)
        t = ((w[:, None] * lb).sum(0) / w.sum()).clone().requires_grad_(True)
        v = fit([V, c, t], lambda: (w * (sb * (lb - torch.log_softmax(t + c @ V.T, 1))).sum(1)).sum(), ITERS)
        if v < best[0]:
            best = (v, (t + c @ V.T).detach())
    return best


out = dict(rows=[])
for net in ("resnet18", "resnet50"):
    meta = np.load(os.path.join(HERE, f"cache/map/{net}_meta.npz"))
    Y = torch.from_numpy(np.asarray(np.load(os.path.join(HERE, f"cache/map/{net}_y.npy"), mmap_mode="r")).astype(np.float64))
    lab = torch.from_numpy(meta["labels"].astype(np.int64))
    S = torch.softmax(Y, 1)
    sb = torch.stack([S[lab == c].mean(0) for c in range(1000)]).clamp_min(1e-12)
    sb = sb / sb.sum(1, keepdim=True)
    W = meta["W"].astype(np.float64)
    Wc = W - W.mean(0)
    Wc = Wc / np.linalg.norm(Wc, axis=1, keepdims=True)
    Lk = linkage(Wc, "ward")
    pi = torch.ones(1000) / 1000
    for k in ((14,) if a.smoke else (14, 38)):
        torch.manual_seed(k)
        whole, L = table_fit(sb, pi, k)
        q = torch.softmax(L, 1)
        for J in ((2, 5) if a.smoke else (2, 5, 10, 20)):
            lab_c = fcluster(Lk, J, "maxclust") - 1
            for kind in ("clusters", "random"):
                part = lab_c if kind == "clusters" else rng.permutation(lab_c)
                total = 0.
                sizes = []
                for b in range(part.max() + 1):
                    H = torch.from_numpy(np.where(part == b)[0])
                    if len(H) < 2:
                        continue
                    sizes.append(len(H))
                    mass = sb[:, H].sum(1)
                    w = pi * mass
                    keep = w > 1e-9
                    sub = sb[keep][:, H]
                    sub = sub / sub.sum(1, keepdim=True)
                    if k + 1 >= len(H):
                        continue  # rank covers it: the least is 0
                    total += table_fit(sub, w[keep], k)[0]
                # between families, at the whole fit
                mat = torch.zeros(1000, part.max() + 1)
                matq = torch.zeros(1000, part.max() + 1)
                pt = torch.from_numpy(part)
                mat.index_add_(1, pt, sb)
                matq.index_add_(1, pt, q)
                between = float((mat * (mat.clamp_min(1e-300).log() - matq.clamp_min(1e-300).log())).sum(1).mean())
                row = dict(net=net, k=k, J=J, kind=kind, whole=whole, minors_sum=total, share=total / whole, between=between, between_share=between / whole, sizes=sorted(sizes)[-5:])
                out["rows"].append(row)
                print(row, flush=True)
    if not a.smoke:
        json.dump(out, open(os.path.join(HERE, "results_pc_real_minors.json"), "w"), indent=1)
print("done", flush=True)
