# serves: DQ15
"""Part C, Conjecture DQ15's route step 5: measure its numbers on real fits, before any proof. cache/map, 20000 crops.
Estimator, fixed before the run. Nets: ResNet-18, ResNet-50. Free codes on the class-row table at k = 14, 38, and the always-on read at k = 14. Fit: L-BFGS 1500 iterations (table), 800 (read), float64. H_x = the top m classes of s_x, for m = k + 1, 2 (k + 1), 100. Reported for each: the mass outside H_x (mean zeta); Tail as a share of D; g_H; rho_x (median and least); sigma_H (on the 200 rows of largest weight in rho terms, a sample); the premise ratio g_H / min over x of min(rho_x, rho_x sigma_H); and the same with the row-wise size, the median over x of |row of P_Z G on H_x| / rho_x, which says how far a typical row is from the row-wise test of Theorem TQ18 part 6.
Prediction, written before the run: the premise ratio falls by many orders against the full test (1e23 to 1e78), and stays above 10. The typical row is above 1, because what a row leaves undone on its head is of the size of its head probabilities. Tail is under a tenth of D at m = 100.
Usage: pc_real_head.py [--smoke]"""
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
torch.set_num_threads(6)
torch.set_default_dtype(torch.float64)
IT_T, IT_R = (30, 20) if a.smoke else (1500, 800)


def fit(params, fn, iters):
    opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, history_size=30, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-13)

    def cl():
        opt.zero_grad()
        v = fn()
        v.backward()
        for p in params:
            p.grad = p.grad.contiguous()
        return v
    opt.step(cl)
    return float(fn().detach())


def study(tag, S, p, t, V, c, Qz, k):
    mean = c.mean(0, keepdim=True)
    t2 = t + (mean @ V.T)[0]
    c2 = c - mean
    Vc = V - V.mean(0, keepdim=True)
    Q, Rr = torch.linalg.qr(Vc)
    c2 = c2 @ Rr.T
    q = torch.softmax(t2 + c2 @ Q.T, 1)
    G = p[:, None] * (S - q)
    PG = G if Qz is None else Qz @ (Qz.T @ G)
    sigma = float(torch.linalg.svdvals(c2).min())
    D = float((p * (S * (S.clamp_min(1e-300).log() - q.log())).sum(1)).sum())
    n, C = S.shape
    order = torch.argsort(S, 1, descending=True)
    out = []
    for m in (k + 1, 2 * (k + 1), 100):
        H = order[:, :m]
        mask = torch.zeros_like(S, dtype=torch.bool).scatter_(1, H, True)
        zeta = (S * (~mask)).sum(1)
        qH = torch.where(mask, q, torch.full_like(q, float("inf"))).min(1).values
        rho_x = p * qH
        PGH = PG * mask
        gH = float(torch.linalg.matrix_norm(PGH, 2))
        rowsize = PGH.norm(dim=1)
        ratio_row = rowsize / rho_x
        idx = torch.argsort(rho_x, descending=True)[:: max(1, n // 200)][:200]
        smin = []
        for i in idx.tolist():
            VH = Q[H[i]]
            VH = VH - VH.mean(0, keepdim=True)
            sv = torch.linalg.svdvals(VH)
            smin.append(float(sv[-1]) if len(sv) >= k else 0.)
        sigma_H = sigma * min(smin)
        sigma_H_med = sigma * float(np.median(smin))
        floor = torch.minimum(rho_x, rho_x * sigma_H)
        floor_med = torch.minimum(rho_x, rho_x * sigma_H_med)
        # Tail: the sum of p_x zeta_x KL(w_x || normalised tail of q_x)
        sT = S * (~mask)
        qT = q * (~mask)
        w = sT / sT.sum(1, keepdim=True).clamp_min(1e-300)
        u = qT / qT.sum(1, keepdim=True).clamp_min(1e-300)
        kl_t = (w * (w.clamp_min(1e-300).log() - u.clamp_min(1e-300).log()))
        kl_t = torch.where(~mask, kl_t, torch.zeros_like(kl_t)).sum(1)
        tail = float((p * zeta * kl_t).sum())
        row = dict(case=tag, k=k, m=m, D=D, zeta_mean=float((p * zeta).sum()), tail=tail, tail_share=tail / D, g_H=gH, sigma=sigma, sigma_H=sigma_H, sigma_H_median=sigma_H_med, rho_median=float(rho_x.median()), rho_least=float(rho_x.min()), premise_ratio=gH / float(floor.min()) if float(floor.min()) > 0 else None, premise_ratio_median_row=gH / float(floor_med.median()), row_size_over_rho_median=float(ratio_row.median()), row_size_over_rho_90=float(ratio_row.quantile(.9)))
        out.append(row)
        print(row, flush=True)
    return out


out = []
for net in ("resnet18", "resnet50"):
    Z = np.load(os.path.join(HERE, f"cache/map/{net}_z.npy"), mmap_mode="r")
    Y = np.load(os.path.join(HERE, f"cache/map/{net}_y.npy"), mmap_mode="r")
    lab = torch.from_numpy(np.load(os.path.join(HERE, f"cache/map/{net}_meta.npz"))["labels"].astype(np.int64))
    n = 1500 if a.smoke else Z.shape[0]
    Yt = torch.from_numpy(np.asarray(Y[:n]).astype(np.float64))
    S = torch.softmax(Yt, 1)
    ids = torch.unique(lab[:n])
    sb = torch.stack([S[lab[:n] == c].mean(0) for c in ids]).clamp_min(1e-12)
    sb = sb / sb.sum(1, keepdim=True)
    lb = sb.log()
    pi = torch.ones(len(ids)) / len(ids)
    for k in ((14,) if a.smoke else (14, 38)):
        torch.manual_seed(k)
        V = (torch.randn(1000, k) * .3).requires_grad_(True)
        c = (torch.randn(len(ids), k) * .3).requires_grad_(True)
        t = lb.mean(0).clone().requires_grad_(True)
        fit([V, c, t], lambda: (pi * (sb * (lb - torch.log_softmax(t + c @ V.T, 1))).sum(1)).sum(), IT_T)
        out += study(net + " table", sb, pi, t.detach(), V.detach(), c.detach(), None, k)
    Zt = torch.from_numpy(np.asarray(Z[:n]).astype(np.float64))
    Zt = (Zt - Zt.mean(0)) / (Zt.std(0) + 1e-4)
    D_ = Yt - Yt.mean(1, keepdim=True)
    ybar = D_.mean(0)
    D_ = D_ - ybar
    U = torch.linalg.eigh(D_.T @ D_ / n)[1].flip(1)
    k = 14
    Q0 = U[:, :k].contiguous()
    M = torch.linalg.lstsq(Zt, D_ @ Q0).solution.contiguous().requires_grad_(True)
    V = Q0.clone().requires_grad_(True)
    t = ybar.clone().requires_grad_(True)
    ls = torch.log_softmax(Yt, 1)
    fit([V, M, t], lambda: (S * (ls - torch.log_softmax(t + (Zt @ M) @ V.T, 1))).sum(1).mean(), IT_R)
    Qz = torch.linalg.qr(torch.cat([torch.ones(n, 1), Zt], 1))[0]
    out += study(net + " read", S, torch.ones(n) / n, t.detach(), V.detach(), (Zt @ M).detach(), Qz, k)
    if not a.smoke:
        json.dump(out, open(os.path.join(HERE, "results_pc_real_head.json"), "w"), indent=1)
print("done", flush=True)
