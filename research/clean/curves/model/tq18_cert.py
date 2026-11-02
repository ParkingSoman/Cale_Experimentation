# serves: TQ18
"""Checks of Theorem TQ18. Estimators fixed before the run. float64.
Instances are built backward. Take a code class (free codes, or a read of a random z of d features), a random rank-k triple (t, V, c), and a random matrix projected to meet the stationarity conditions, scaled so g is a chosen fraction of the test. s_x = q_x + G_x / p_x.
A. Algebra at random competitors: (F) |E|_F^2 = 2 X, |E|_op <= 1, |R|_op = max(g, w/sigma), G° feasible and q° a distribution, weak duality D(competitor) >= dual value, and dual value >= entropy sum + X g (sigma - 2 g / rho). Fails if any is off by > 1e-9.
B. Part 1. Sizes (rows, classes, k) in {(6, 6, 1), (6, 6, 2), (8, 8, 3), (10, 6, 2), (12, 8, 3), (20, 10, 4)}, free codes and read (d = rows - 2), scale 0.5 and 1.0 of the test. 15 random starts per instance, L-BFGS, against the entropy sum. Prediction, written before the run: no start below it by more than 1e-9.
C. Part 3, graded, and how tight. Same builder at scale 2, 5, 20, 100 of the test, keeping g < rho. Fit from 15 starts. Reported: bound = k g (2 g/rho - sigma)_+, true gap = D - least found (above 0 only if a fit beats the point), and bound / gap where gap > 1e-9. Fails if gap > bound + 1e-9.
D. Part 4: symmetric positive semidefinite L of rank k with K not equal to 2k. Checked as B.
Usage: tq18_cert.py [--smoke] [--threads N]"""
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
torch.set_default_dtype(torch.float64)
torch.set_num_threads(a.threads)
torch.manual_seed(18)
rng = np.random.default_rng(18)
STARTS = 5 if a.smoke else 15


def entropy(q):
    return -(q * np.log(q)).sum(-1)


def softmax(l):
    l = l - l.max(1, keepdims=True)
    q = np.exp(l)
    return q / q.sum(1, keepdims=True)


def proj_space(Zf):
    """Orthogonal projector onto the span of the columns of Zf."""
    Q, s, _ = np.linalg.svd(Zf, full_matrices=False)
    Q = Q[:, s > 1e-10 * s[0]]
    return Q @ Q.T


def build(n, C, k, d, scale, mode="free"):
    """A rank-k triple with a residual that meets stationarity, scaled to `scale` times the test."""
    if mode == "free":
        Z = np.eye(n)
    else:
        Z = np.column_stack([np.ones(n), rng.normal(size=(n, d))])
    Pz = proj_space(Z)
    p = rng.dirichlet(np.ones(n) * 30.)
    Q, _ = np.linalg.qr(np.column_stack([np.ones(C), rng.normal(size=(C, k))]))
    V = Q[:, 1:k + 1]
    c0 = rng.normal(size=(n, k)) * rng.uniform(.3, 1.2)
    c = Pz @ c0
    c = c - c.mean(0, keepdims=True)  # still in Z, because Z holds 1
    t = rng.normal(size=C) * .4
    q = softmax(t + c @ V.T)
    u, sv, vt = np.linalg.svd(c, full_matrices=False)
    sigma = sv.min()
    rho = (p[:, None] * q).min()
    Pc = u @ u.T
    PiV = np.eye(C) - V @ V.T - np.ones((C, C)) / C
    In = np.eye(n)
    one = np.ones((n, 1)) / np.sqrt(n)
    P1 = one @ one.T
    # the explained part: rows in the complement of V and 1; left side in Z, orthogonal to 1 and to c
    Lz = Pz - P1 - Pc
    A = rng.normal(size=(n, C))
    G1 = Lz @ A @ PiV
    # the unexplained part: left side orthogonal to Z, rows summing to 0
    Lu = In - Pz
    J = np.eye(C) - np.ones((C, C)) / C
    G2 = Lu @ rng.normal(size=(n, C)) @ J
    test = rho * min(1.0, sigma / 2)
    gn = np.linalg.norm(G1, 2)
    target = scale * test if scale <= 1 else min(scale * rho * sigma / 2, .95 * rho)
    if gn < 1e-14:
        return None
    G1 = G1 * (target / gn)
    # scale the unexplained part so that s stays positive
    ratio = np.abs((G2 / p[:, None]) / q).max() if np.abs(G2).max() > 0 else 0.
    G2 = G2 * (min(.5, rng.uniform(.1, 2) * target * 50) / ratio if ratio > 0 else 0.)
    G = G1 + G2
    s = q + G / p[:, None]
    if s.min() <= 0:
        return None
    return dict(s=s, p=p, Z=Z, Pz=Pz, t=t, V=V, c=c, q=q, G=G, g=np.linalg.norm(Pz @ G, 2), sigma=sigma, rho=rho, mode=mode, n=n, C=C, k=k)


def kl_rows(s, l):
    return (s * (s.log() - torch.log_softmax(l, 1))).sum(1)


def fit(inst, starts):
    s = torch.tensor(inst["s"])
    p = torch.tensor(inst["p"])
    n, C, k = inst["n"], inst["C"], inst["k"]
    Zb = torch.tensor(np.linalg.svd(inst["Z"], full_matrices=False)[0][:, np.linalg.svd(inst["Z"], compute_uv=False) > 1e-10 * np.linalg.svd(inst["Z"], compute_uv=False)[0]])
    best = 1e9
    for j in range(starts):
        V = (torch.randn(C, k) * .8).requires_grad_(True)
        C_ = (torch.randn(Zb.shape[1], k) * .8).requires_grad_(True)
        t = torch.zeros(C).requires_grad_(True)
        opt = torch.optim.LBFGS([V, C_, t], lr=1, max_iter=3000, history_size=60, line_search_fn="strong_wolfe", tolerance_grad=1e-11, tolerance_change=1e-16)

        def cl():
            opt.zero_grad()
            v = (p * kl_rows(s, t + (Zb @ C_) @ V.T)).sum()
            v.backward()
            return v
        opt.step(cl)
        best = min(best, float((p * kl_rows(s, t + (Zb @ C_) @ V.T)).sum().detach()))
    return best


def value(inst):
    return float((inst["p"] * (entropy(inst["q"]) - entropy(inst["s"]))).sum())


def check_algebra(inst, trials):
    n, C, k = inst["n"], inst["C"], inst["k"]
    s, p, V, c, G, Pz, q = inst["s"], inst["p"], inst["V"], inst["c"], inst["G"], inst["Pz"], inst["q"]
    g, sigma, rho = inst["g"], inst["sigma"], inst["rho"]
    B = c @ np.linalg.inv(c.T @ c)
    w = g * sigma
    R = Pz @ G + w * B @ V.T
    fails = 0
    fails += abs(np.linalg.norm(R, 2) - max(g, w / sigma)) > 1e-9
    base = float((p * (entropy(q) - entropy(s))).sum())
    Pi0 = np.eye(C) - V @ V.T - np.ones((C, C)) / C
    for _ in range(trials):
        Vp = rng.normal(size=(C, k))
        Qx, _ = np.linalg.qr(np.column_stack([np.ones(C), Vp]))
        Pi = np.eye(C) - Qx @ Qx.T
        E = Pi - Pi0
        X = np.trace(V.T @ Pi @ V)
        fails += abs((E ** 2).sum() - 2 * X) > 1e-9
        fails += np.linalg.norm(E, 2) > 1 + 1e-9
        Go = (np.eye(n) - Pz) @ G + R @ Pi
        fails += np.abs(Go - (G + R @ E)).max() > 1e-9
        qo = s - Go / p[:, None]
        if g < rho:
            fails += qo.min() < -1e-12
            fails += np.abs(qo.sum(1) - 1).max() > 1e-9
            fails += np.abs(Go.sum(0)).max() > 1e-9
            cp = Pz @ rng.normal(size=(n, k))
            fails += np.abs(cp.T @ Go @ Vp).max() > 1e-8
            tp = rng.normal(size=C)
            lp = tp + cp @ Vp.T
            D = float((p * (s * (np.log(s) - np.log(softmax(lp)))).sum(1)).sum())
            dual = float((p * (entropy(np.maximum(qo, 1e-300)) - entropy(s))).sum())
            fails += D < dual - 1e-9
            fails += dual < base + X * g * (sigma - 2 * g / rho) - 1e-9
    return int(fails)


out = {}
t0 = time.time()
SIZES = [(6, 6, 1), (8, 8, 2)] if a.smoke else [(6, 6, 1), (6, 6, 2), (8, 8, 3), (10, 6, 2), (12, 8, 3), (20, 10, 4)]
# A and B
rowsB, fa, fb = [], 0, 0
for (n, C, k) in SIZES:
    for mode in ("free", "read"):
        if mode == "read" and n - 2 <= k:
            continue
        for scale in (.5, 1.0):
            inst = None
            while inst is None:
                inst = build(n, C, k, n - 2, scale, mode)
            fa += check_algebra(inst, 50 if a.smoke else 300)
            val = value(inst)
            best = fit(inst, STARTS)
            row = dict(n=n, C=C, k=k, mode=mode, scale=scale, g=inst["g"], sigma=inst["sigma"], rho=inst["rho"], test=inst["rho"] * min(1, inst["sigma"] / 2), value=val, least_found=best, below=val - best)
            fb += row["below"] > 1e-9
            rowsB.append(row)
            print(row, flush=True)
out["A"] = dict(fails=fa)
out["B"] = dict(rows=rowsB, fails=fb)
print("A fails", fa, "B fails", fb, "of", len(rowsB), flush=True)
# C
rowsC, fc = [], 0
for (n, C, k) in SIZES:
    for mode in ("free", "read"):
        if mode == "read" and n - 2 <= k:
            continue
        for scale in (2, 5, 20, 100):
            inst = None
            while inst is None:
                inst = build(n, C, k, n - 2, scale, mode)
            g, sigma, rho = inst["g"], inst["sigma"], inst["rho"]
            val = value(inst)
            best = fit(inst, STARTS)
            gap = val - best
            bnd = k * g * max(2 * g / rho - sigma, 0.)
            row = dict(n=n, C=C, k=k, mode=mode, scale=scale, g=g, sigma=sigma, rho=rho, value=val, least_found=best, gap=gap, bound=bnd, ratio=bnd / gap if gap > 1e-9 else None)
            fc += gap > bnd + 1e-9
            rowsC.append(row)
            print(row, flush=True)
out["C"] = dict(rows=rowsC, fails=fc)
print("C fails", fc, flush=True)
# D
rowsD, fd = [], 0
for (K, k) in ([(6, 1), (8, 3)] if a.smoke else [(6, 1), (8, 1), (8, 3), (10, 2), (12, 3), (12, 5)]):
    for trial in range(1 if a.smoke else 3):
        Q, _ = np.linalg.qr(np.column_stack([np.ones(K), rng.normal(size=(K, k))]))
        U = Q[:, 1:k + 1]
        lam = rng.uniform(.5, 3, size=k)
        L = (U * lam) @ U.T
        pi = rng.dirichlet(np.ones(K) * 30.)
        t = rng.normal(size=K) * .3
        q = softmax(t + L)
        rho = (pi[:, None] * q).min()
        PiL = np.eye(K) - U @ U.T - np.ones((K, K)) / K
        sigma = lam.min()
        mu = .9 * rho * min(1, sigma / 2)
        s = q + mu * PiL / pi[:, None]
        inst = dict(s=s, p=pi, Z=np.eye(K), n=K, C=K, k=k)
        val = float((pi * (entropy(q) - entropy(s))).sum())
        best = fit(inst, STARTS)
        row = dict(K=K, k=k, mu=mu, rho=rho, sigma=sigma, value=val, least_found=best, below=val - best)
        fd += row["below"] > 1e-9
        rowsD.append(row)
        print(row, flush=True)
out["D"] = dict(rows=rowsD, fails=fd)
out["sec"] = time.time() - t0
if not a.smoke:
    json.dump(out, open(os.path.join(HERE, "tq18_cert.json"), "w"), indent=1, default=float)
print("D fails", fd, "done", out["sec"], flush=True)
