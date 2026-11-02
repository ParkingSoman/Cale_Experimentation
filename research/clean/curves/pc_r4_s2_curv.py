# serves: DQ23
"""Round 4, screen S2 (card 3-37): is the read's curvature near a Kronecker product? It is the one candidate that sees the encoder.

Estimator, fixed before the run. ResNet-18 and -50, cache/map z and y, float64, z whitened. K = 1 model: Hbar = Fbar kron I, with
Fbar the mean of F_x (Definition DQ23; Sigma = I after whitening). lambda_max and lambda_min of Hbar^(-1/2) H Hbar^(-1/2), by the
Lanczos method on the range of Hbar (Fbar has a null direction, the constant vector), 60 steps.
Also the two numbers that matter for a floor and a rank-k read, at the cached read fit W_fit (rank 14, 38):
  the ratio <W, H W> / <W, Hbar W> at W = the part of the teacher's full read that the fit leaves undone;
  the same ratio on the fit's own span directions.
K > 1: the same test with regions = N's predicted class, by power iteration on Hbar_K^(-1) H, with Hbar_K applied by conjugate
gradients, only if the K = 1 result does not already pass.

Screen line, fixed before the run: PASS if |log lambda| <= log 2, about .69, in both directions. That is, lambda in [.5, 2].

PREDICTION, written before the first run (2026-11-02): FAIL by a lot at K = 1: lambda_max above 10, since one class's direction
is stiff on its own photos and soft elsewhere (TQ7's eta = 1 - 1e-8 is the entrywise form of this). lambda_min below .1. The ratio at
the undone part is within a factor 2 of 1, since it is an average over many directions.

Usage: pc_r4_s2_curv.py [--smoke] [--n 8000]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--n", type=int, default=8000)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)
STEPS = 8 if a.smoke else 60
out = {}
for net in ("resnet18", "resnet50"):
    Zm = np.load(os.path.join(HERE, f"cache/map/{net}_z.npy"), mmap_mode="r")
    Ym = np.load(os.path.join(HERE, f"cache/map/{net}_y.npy"), mmap_mode="r")
    n = 800 if a.smoke else min(a.n, Zm.shape[0])
    Z = torch.from_numpy(np.asarray(Zm[:n]).astype(np.float64))
    Y = torch.from_numpy(np.asarray(Ym[:n]).astype(np.float64))
    Z = Z - Z.mean(0)
    # whiten on the range of z
    ev, U = torch.linalg.eigh(Z.T @ Z / n)
    keep = ev > ev.max() * 1e-8
    Zw = Z @ U[:, keep] / ev[keep].sqrt()
    d = Zw.shape[1]
    S = torch.softmax(Y, 1)
    C = S.shape[1]
    Fbar = torch.diag(S.mean(0)) - S.T @ S / n
    fe, fU = torch.linalg.eigh(Fbar)
    fkeep = fe > fe.max() * 1e-10
    fi = fU[:, fkeep] / fe[fkeep].sqrt()  # C x r, Fbar^(-1/2) on its range
    f_h = fU[:, fkeep] * fe[fkeep].sqrt()
    r = fi.shape[1]

    def Hq(W):
        """(W: C x d)  ->  H W, in the same shape."""
        L = Zw @ W.T  # n x C
        m = (S * L).sum(1, keepdim=True)
        G = S * (L - m)  # n x C: F_x L_x
        return (G.T @ Zw) / n

    def A(X):
        """Hbar^(-1/2) H Hbar^(-1/2) on r x d matrices. Hbar = Fbar kron I, so its root acts on the class side only."""
        return fi.T @ Hq(fi @ X)

    def lanczos(steps, seed):
        torch.manual_seed(seed)
        v = torch.randn(r, d)
        v = v / v.norm()
        vs, al, be = [], [], []
        w_prev = None
        beta = 0.
        for j in range(steps):
            w = A(v)
            alpha = float((w * v).sum())
            w = w - alpha * v - (beta * w_prev if w_prev is not None else 0)
            for u in vs:  # full reorthogonalisation: the extreme values must be real
                w = w - (w * u).sum() * u
            vs.append(v)
            al.append(alpha)
            beta = float(w.norm())
            be.append(beta)
            w_prev = v
            v = w / beta
        T = torch.diag(torch.tensor(al)) + torch.diag(torch.tensor(be[:-1]), 1) + torch.diag(torch.tensor(be[:-1]), -1)
        return torch.linalg.eigvalsh(T)

    t0 = time.time()
    lam = lanczos(STEPS, 1)
    row = dict(n=n, d=d, r=r, lam_max=float(lam.max()), lam_min=float(lam.min()), lam_top5=[float(x) for x in lam[-5:]], lam_low5=[float(x) for x in lam[:5]], sec=time.time() - t0)
    print(net, row, flush=True)
    # the teacher's full read, in whitened z, and the part a rank-k read leaves undone, as directions in W
    Yc = Y - Y.mean(1, keepdim=True)
    Yc = Yc - Yc.mean(0)
    Wfull = torch.linalg.lstsq(Zw, Yc).solution.T  # C x d
    row["full_read_spread"] = float((Zw @ Wfull.T - Yc).norm() / Yc.norm())
    sv_U, sv_S, sv_V = torch.linalg.svd(Wfull, full_matrices=False)
    for k in (14, 38):
        for name, W in (("undone", (sv_U[:, k:] * sv_S[k:]) @ sv_V[k:]), ("kept", (sv_U[:, :k] * sv_S[:k]) @ sv_V[:k])):
            q = float((W * Hq(W)).sum())
            qb = float((Fbar @ W * W).sum())
            row[f"ratio_{name}_{k}"] = q / qb
    print(net, {x: row[x] for x in row if x.startswith("ratio") or x == "full_read_spread"}, flush=True)
    out[net] = row
if not a.smoke:
    with open(os.path.join(HERE, "rounds/round4/s2_curv.json"), "w") as f:
        json.dump(out, f, indent=1)
