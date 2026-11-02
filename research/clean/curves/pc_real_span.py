# serves: TQ11
"""Part C, the real object of Definition DQ9 on ResNet-18 (35000 ImageNet validation crops), one read. What is the span?
Estimator, fixed before the run. k = 14 and 38. Every value is the KL in nats at the end of L-BFGS from double precision.
 (a) joint search over (V, M, t) from four starts: logit PCA, probability PCA, Gamma (below), random. Do they end at one value?
 (b) fixed-span convex fits (M, t free, V fixed) at closed-form spans:
     logit: top-k eigenvectors of Cov(centred logits)
     prob:  top-k eigenvectors of Cov(softmax)
     fisher: B^(-1/2) U_k, U_k top eigenvectors of B^(1/2) Cov(logits) B^(1/2), B the mean Fisher metric = diag(sbar) - E[s s^T]
     gamma: top-k left singular vectors of Gamma = E[(s - sbar) z^T], z the whitened channels (reduced-rank fit of s on h;
            also the first step of Frank-Wolfe from k = 0, since Gamma is the gradient of D at the mean softmax)
 (c) nesting: principal angles between the best span at k = 14 and the best span at k = 38.
Prediction, written before the run: gamma and prob are far better than logit (the moments of a KL fit live in probabilities, not
logits); the joint search ends within 10% of its best from every start.
Usage: pc_real_span.py [--smoke]"""
import argparse
import json
import os
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--iters", type=int, default=400)
ap.add_argument("--fix-iters", type=int, default=150)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(8)
torch.manual_seed(1101)
Yn = np.load(os.path.join(HERE, "cache/r18/y_train.npy"), mmap_mode="r")
Hn = np.load(os.path.join(HERE, "cache/r18/h_train.npy"), mmap_mode="r")
n = 2000 if a.smoke else Yn.shape[0]
Y = torch.tensor(np.asarray(Yn[:n]), dtype=torch.float64)
H = torch.tensor(np.asarray(Hn[:n]), dtype=torch.float64)
C = Y.shape[1]
ls = torch.log_softmax(Y, 1)
S = ls.exp()
sbar = S.mean(0)
Hc = H - H.mean(0)
ev_h, U_h = torch.linalg.eigh(Hc.T @ Hc / n)
keep = ev_h > 1e-8 * ev_h.max()
Zw = Hc @ U_h[:, keep] @ torch.diag(ev_h[keep] ** -0.5)   # whitened channels: the rank constraint does not see this change
Yc = Y - Y.mean(1, keepdim=True)
ybar = Yc.mean(0)
D = Yc - ybar
P = torch.eye(C, dtype=torch.float64) - torch.ones(C, C, dtype=torch.float64) / C


def kl(L):
    return (S * (ls - torch.log_softmax(L, 1))).sum(1).mean()


def fit(params, fn, iters):
    opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, history_size=30, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-12)

    def cl():
        opt.zero_grad()
        v = fn()
        v.backward()
        for p in params:
            p.grad = p.grad.contiguous()
        return v
    opt.step(cl)
    return float(fn().detach())


def top(A, k):
    ev, U = torch.linalg.eigh((A + A.T) / 2)
    return U[:, ev.argsort(descending=True)[:k]].contiguous().clone()


CovY = D.T @ D / n
Sc = S - sbar
CovS = P @ (Sc.T @ Sc / n) @ P
B = torch.diag(sbar) - S.T @ S / n
ev, Ub = torch.linalg.eigh(P @ B @ P)
good = ev > 1e-10 * ev.max()
Bh = Ub[:, good] @ torch.diag(ev[good] ** .5) @ Ub[:, good].T
Bi = Ub[:, good] @ torch.diag(ev[good] ** -.5) @ Ub[:, good].T
Gamma = P @ (Sc.T @ Zw / n)
U_g, sv_g, _ = torch.linalg.svd(Gamma, full_matrices=False)


def span(kind, k):
    if kind == "logit":
        return top(CovY, k)
    if kind == "prob":
        return top(CovS, k)
    if kind == "fisher":
        return torch.linalg.qr(Bi @ top(Bh @ CovY @ Bh, k))[0]
    if kind == "gamma":
        return U_g[:, :k].contiguous().clone()
    if kind == "random":
        return torch.linalg.qr(P @ torch.randn(C, k, dtype=torch.float64))[0]


def read0(V):
    return torch.linalg.lstsq(Zw, D @ V).solution.contiguous().clone()   # dims x k


out = {"n": n, "rows": []}
best = {}
for k in (14, 38):
    for kind in ("logit", "prob", "fisher", "gamma"):
        t0 = time.time()
        V = span(kind, k)
        M = read0(V).requires_grad_(True)
        t = ybar.contiguous().clone().requires_grad_(True)
        val = fit([M, t], lambda: kl(t + (Zw @ M) @ V.T), a.fix_iters)
        row = dict(k=k, what="fixed span", span=kind, kl=val, sec=time.time() - t0)
        out["rows"].append(row)
        print(row, flush=True)
    for kind in ("logit", "prob", "gamma", "random"):
        t0 = time.time()
        V = span(kind, k).requires_grad_(True)
        M = read0(V.detach()).requires_grad_(True)
        t = ybar.contiguous().clone().requires_grad_(True)
        val = fit([V, M, t], lambda: kl(t + (Zw @ M) @ V.T), a.iters)
        row = dict(k=k, what="joint search", start=kind, kl=val, sec=time.time() - t0,
                   nuc=float(torch.linalg.svdvals((V @ M.T).detach()).sum()))
        out["rows"].append(row)
        print(row, flush=True)
        if k not in best or val < best[k][0]:
            best[k] = (val, torch.linalg.qr(V.detach())[0], (V @ M.T).detach())
    json.dump(out, open(os.path.join(HERE, "results_pc_real_span.json"), "w"), indent=1)
cos = torch.linalg.svdvals(best[14][1].T @ best[38][1])
out["nesting_cos_14_in_38"] = [float(x) for x in cos]
print("cosines of the principal angles, best span k=14 inside best span k=38:", [round(float(x), 3) for x in cos], flush=True)
for kind in ("logit", "prob", "fisher", "gamma"):
    c = torch.linalg.svdvals(span(kind, 38).T @ best[38][1])
    out[f"cos_{kind}_38"] = [float(x) for x in c]
    print(kind, "cosines with the best span k=38: mean", round(float(c.mean()), 3), "min", round(float(c.min()), 3), flush=True)
sv = torch.linalg.svdvals(best[38][2])
out["sv_best_38_top"] = [float(x) for x in sv[:40]]
A = torch.linalg.lstsq(Zw, D).solution      # N's own head in the whitened channels
svN = torch.linalg.svdvals(A)
for k in (14, 38):
    out[f"sizes_{k}"] = dict(R_best=float(best[k][2].norm()), nuc_best=float(torch.linalg.svdvals(best[k][2]).sum()),
                             R_N=float(svN.norm()), nuc_N=float(svN.sum()), R_N_topk=float(svN[:k].norm()))
    print(k, out[f"sizes_{k}"], flush=True)
json.dump(out, open(os.path.join(HERE, "results_pc_real_span.json"), "w"), indent=1)
np.savez(os.path.join(HERE, "cache/map/r18_best_spans.npz"), V14=best[14][1].numpy(), V38=best[38][1].numpy())
print("done", flush=True)
