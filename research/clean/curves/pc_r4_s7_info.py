# serves: TQ11
"""Round 4, S7 (card 3-37): an information floor for the read, and a closed-form encoder. It uses the encoder, unlike any floor through free codes.

Idea. Let i be drawn from s_x. For a law on x (not a finite list, where a generic projection is one-to-one), any decoder q from y = M z(x) has
   E KL(s_x || q(. | y)) >= I(i; x) - I(i; y).
So a floor needs an upper bound on I(i; y) over all k-dim M. A Gaussian bound with one within covariance gives I(i; y) <= (1/2) sum over j <= k of log(1 + lambda_j), with
lambda_j the eigenvalues of Sigma_w^(-1) Sigma_b, for i drawn from s_x. So F(k) = I(i; x) - (1/2) sum over j <= k of log(1 + lambda_j), only as a screen.

Estimator, fixed before the run. cache/map. float64. Sigma_b from class means with weights s_x[i].
Build: M = the top k eigenvectors, then a convex fit of (V, t), 400 iterations. A closed-form encoder with one convex fit, label ONE-BUILD.

Screen line, fixed before the run: F(k) >= (read fit) / 2 at k = 14, 38, on both nets.
PREDICTION, written before the first run (2026-11-02): F(k) is negative at k = 14 and 38: 14 dimensions hold more Gaussian capacity than the 5 to 6 nats of I(i; x), so the bound is empty.
The build's loss is within 1.5 times the read fit.

Usage: pc_r4_s7_info.py [--smoke]
"""
import argparse
import json
import os

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(3)
torch.set_default_dtype(torch.float64)
ITERS = 20 if a.smoke else 400
out = {}
for net in ("resnet18", "resnet50"):
    Zm = np.load(os.path.join(HERE, f"cache/map/{net}_z.npy"), mmap_mode="r")
    Ym = np.load(os.path.join(HERE, f"cache/map/{net}_y.npy"), mmap_mode="r")
    n = 1500 if a.smoke else Zm.shape[0]
    Z = torch.from_numpy(np.asarray(Zm[:n]).astype(np.float64))
    S = torch.softmax(torch.from_numpy(np.asarray(Ym[:n]).astype(np.float64)), 1)
    T = S.clamp_min(1e-300).log()
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-4)
    pi = S.mean(0)
    Hbar = -(pi * pi.clamp_min(1e-300).log()).sum()
    Hx = -(S * T).sum(1).mean()
    I_total = float(Hbar - Hx)
    mu = (S.T @ Z) / n / pi[:, None].clamp_min(1e-300)  # C x d: mean of z for i ~ s_x
    Sb = (mu * pi[:, None]).T @ mu
    St = Z.T @ Z / n
    ev, U = torch.linalg.eigh(St + 1e-6 * torch.eye(St.shape[0]))
    W_ = U / ev.sqrt()
    evb, Ub = torch.linalg.eigh(W_.T @ Sb @ W_)
    lam_t = evb.flip(0).clamp(0, 1 - 1e-9)
    lam = lam_t / (1 - lam_t)  # eigenvalues of Sw^-1 Sb
    dirs = W_ @ Ub.flip(1)
    row = dict(n=n, d=Z.shape[1], I_total=I_total, Hbar=float(Hbar), Hx=float(Hx), lam_top=[float(x) for x in lam[:5]], lam_14=float(lam[13]), lam_38=float(lam[37]))
    for k in (14, 38):
        cap = float(.5 * torch.log1p(lam[:k]).sum())
        row[f"cap_{k}"] = cap
        row[f"F_{k}"] = I_total - cap
        if not a.smoke:
            f = np.load(os.path.join(HERE, f"cache/map/fit_{net}_read_{k}.npz"))
            Vf, Mf, tf = (torch.from_numpy(f[x].astype(np.float64)) for x in ("V", "M", "t"))
            with torch.no_grad():
                row[f"read_fit_{k}"] = float((S * (T - torch.log_softmax(tf + (Z @ Mf) @ Vf.T, 1))).sum(1).mean())
        C_ = Z @ dirs[:, :k]
        C_ = C_ / C_.std(0)
        V = torch.zeros(1000, k, requires_grad=True)
        t = T.mean(0).clone().requires_grad_(True)
        o = torch.optim.LBFGS([V, t], lr=1, max_iter=ITERS, history_size=30, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-13)

        def cl():
            o.zero_grad()
            v = (S * (T - torch.log_softmax(t + C_ @ V.T, 1))).sum(1).mean()
            v.backward()
            return v
        o.step(cl)
        row[f"build_{k}"] = float(cl().detach())
    print(net, row, flush=True)
    out[net] = row
if not a.smoke:
    with open(os.path.join(HERE, "rounds/round4/s7_info.json"), "w") as f:
        json.dump(out, f, indent=1)
