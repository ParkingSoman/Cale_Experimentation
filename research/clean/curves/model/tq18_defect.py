# serves: TQ18
"""Checks of the parts added after the referee round. Estimators fixed before the run. float64.
A. Theorem TQ18 part 5 at a triple that is NOT stationary. Build a table with the builder of model/tq18_cert.py at 0.5 of the test, move the triple by a small random step of size e in t, V (re-orthonormalised) and c (kept in Z, centred), and take the moved triple as the point. Sizes (6,6,1), (8,8,2), (12,8,3), free and read, e in {1e-3, 1e-2, 1e-1}. Compute g~, |Delta|_op, d, sigma, rho at the moved triple. When the premise holds, fit from 15 starts; fails if D - least found > the bound + 1e-9. Also the bound on d. Reported: bound / (D - least found).
Prediction: 0 fails. The ratio stays bounded as e -> 0, since both sides go as e^2.
B. Theorem TQ17 part 3, the new sufficient condition: mu_c / m_c + log(1 + mu_c / q'_c) < tau implies g_c > 0, on random draws. Fails if it does not.
C. Theorem TQ17 part 5: at t = 0, every (tau, mu) with premise 2 true has mu_c / 2 <= q'' - q'; on a grid. Fails if not.
D. Theorem TQ17 part 6: tables of the family moved by a small random delta, within its premise, fit from 15 starts. Fails if the loss of the pole build on s~ minus the least found exceeds the bound.
Usage: tq18_defect.py [--smoke]"""
import argparse
import json
import os

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)
torch.manual_seed(21)
rng = np.random.default_rng(21)
STARTS = 5 if a.smoke else 15


def phi(y):
    y = np.asarray(y, dtype=float)
    return np.where(y <= -1, 1.0, (1 + y) * np.log1p(np.maximum(y, -1 + 1e-300)) - y)


def softmax(l):
    l = l - l.max(1, keepdims=True)
    q = np.exp(l)
    return q / q.sum(1, keepdims=True)


def proj(Zf):
    Q, s, _ = np.linalg.svd(Zf, full_matrices=False)
    return Q[:, s > 1e-10 * s[0]]


def fit(s, p, Qz, k, starts):
    s_ = torch.tensor(s)
    p_ = torch.tensor(p)
    Zb = torch.tensor(Qz)
    n, C = s.shape
    best = 1e9
    for j in range(starts):
        V = (torch.randn(C, k) * .8).requires_grad_(True)
        c = (torch.randn(Zb.shape[1], k) * .8).requires_grad_(True)
        t = torch.zeros(C).requires_grad_(True)
        opt = torch.optim.LBFGS([V, c, t], lr=1, max_iter=3000, history_size=60, line_search_fn="strong_wolfe", tolerance_grad=1e-11, tolerance_change=1e-16)

        def cl():
            opt.zero_grad()
            v = (p_ * (s_ * (s_.log() - torch.log_softmax(t + (Zb @ c) @ V.T, 1))).sum(1)).sum()
            v.backward()
            return v
        opt.step(cl)
        best = min(best, float((p_ * (s_ * (s_.log() - torch.log_softmax(t + (Zb @ c) @ V.T, 1))).sum(1)).sum().detach()))
    return best


def build(n, C, k, mode, scale=.5):
    Z = np.eye(n) if mode == "free" else np.column_stack([np.ones(n), rng.normal(size=(n, n - 2))])
    Qz = proj(Z)
    Pz = Qz @ Qz.T
    p = rng.dirichlet(np.ones(n) * 30.)
    Q, _ = np.linalg.qr(np.column_stack([np.ones(C), rng.normal(size=(C, k))]))
    V = Q[:, 1:k + 1]
    c = Pz @ (rng.normal(size=(n, k)) * rng.uniform(.5, 1.2))
    c = c - c.mean(0, keepdims=True)
    t = rng.normal(size=C) * .4
    q = softmax(t + c @ V.T)
    u, sv, _ = np.linalg.svd(c, full_matrices=False)
    rho = (p[:, None] * q).min()
    Pi = np.eye(C) - V @ V.T - np.ones((C, C)) / C
    one = np.ones((n, 1)) / np.sqrt(n)
    G1 = (Pz - one @ one.T - u @ u.T) @ rng.normal(size=(n, C)) @ Pi
    G1 = G1 * (scale * rho * min(1, sv.min() / 2) / np.linalg.norm(G1, 2))
    s = q + G1 / p[:, None]
    return dict(s=s, p=p, Qz=Qz, Pz=Pz, t=t, V=V, c=c, n=n, C=C, k=k) if s.min() > 0 else None


def numbers(inst, t, V, c):
    s, p, Pz = inst["s"], inst["p"], inst["Pz"]
    n, C, k = inst["n"], inst["C"], inst["k"]
    q = softmax(t + c @ V.T)
    G = p[:, None] * (s - q)
    u, sv, _ = np.linalg.svd(c, full_matrices=False)
    one = np.ones((n, 1)) / np.sqrt(n)
    Pw = one @ one.T + u @ u.T
    A = (Pz - Pw) @ G @ (np.eye(C) - V @ V.T)
    Dl = Pz @ G - A
    D = float((p * (s * (np.log(s) - np.log(q))).sum(1)).sum())
    d = np.linalg.norm(Dl)
    dbound = np.linalg.norm(Pz @ G @ V) + 2 * (np.linalg.norm(G.sum(0)) / np.sqrt(n) + np.linalg.norm(c.T @ G) / sv.min())
    return dict(D=D, gt=np.linalg.norm(A, 2), dl_op=np.linalg.norm(Dl, 2), d=d, dbound=dbound, sigma=sv.min(), rho=(p[:, None] * q).min())


out = dict(A=[])
fa = 0
for (n, C, k) in ([(6, 6, 1)] if a.smoke else [(6, 6, 1), (8, 8, 2), (12, 8, 3)]):
    for mode in ("free", "read"):
        for e in (1e-3, 1e-2, 1e-1):
            inst = None
            while inst is None:
                inst = build(n, C, k, mode)
            V2, _ = np.linalg.qr(np.column_stack([np.ones(C), inst["V"] + e * rng.normal(size=inst["V"].shape)]))
            V2 = V2[:, 1:k + 1]
            c2 = inst["c"] + e * inst["Pz"] @ rng.normal(size=inst["c"].shape)
            c2 = c2 - c2.mean(0, keepdims=True)
            t2 = inst["t"] + e * rng.normal(size=C)
            N = numbers(inst, t2, V2, c2)
            kstar = min(k, C - 1 - k)
            prem = N["gt"] + N["dl_op"] <= N["rho"]
            bound = 2 * N["d"] ** 2 / N["rho"] + kstar * N["gt"] * max(4 * N["gt"] / N["rho"] - N["sigma"], 0)
            row = dict(n=n, C=C, k=k, mode=mode, e=e, premise=bool(prem), D=N["D"], bound=bound, d=N["d"], dbound=N["dbound"], dfail=bool(N["d"] > N["dbound"] + 1e-12))
            fa += row["dfail"]
            if prem:
                best = fit(inst["s"], inst["p"], inst["Qz"], k, STARTS)
                gap = N["D"] - best
                row.update(gap=gap, ratio=bound / gap if gap > 1e-12 else None)
                fa += gap > bound + 1e-9
            out["A"].append(row)
            print(row, flush=True)
print("A fails", fa, flush=True)
out["A_fails"] = int(fa)
fb, nb = 0, 0
for _ in range(2000 if a.smoke else 40000):
    K = int(rng.choice([4, 6, 8, 12]))
    q = rng.dirichlet(np.ones(K) * rng.uniform(1, 30))
    i = rng.integers(K)
    qp = q[i]
    m = np.delete(q, i).min()
    tau = rng.uniform(.05, 6)
    mc = rng.uniform(0, 2 * m)
    if mc / m + np.log1p(mc / qp) >= tau:
        continue
    nb += 1
    gam = 4 * m * float(phi(-mc / (2 * m)))
    Ts = np.linspace(1e-6, 1 - 1 / K, 2000)
    g = (mc * tau - gam) * Ts + gam * Ts ** 2 - qp * phi(mc * Ts / qp)
    fb += g.min() <= 0
out["B"] = dict(tested=nb, fails=int(fb))
print("B", out["B"], flush=True)
fc, nc = 0, 0
for K in (4, 6, 8, 12, 20):
    for u in np.geomspace(1.01, 200, 60):
        Z = u + 1 / u + K - 2
        q2, q1 = 1 / Z, 1 / (u * Z)
        for f in np.linspace(.02, 1.98, 99):  # mc / 2 as a share of 2 q''
            mc = f * q2
            gam = 4 * q2 * float(phi(-mc / (2 * q2)))
            Ts = np.linspace(1e-6, 1 - 1 / K, 4000)
            g = (mc * np.log(u) - gam) * Ts + gam * Ts ** 2 - q1 * phi(mc * Ts / q1)
            if g.min() >= -1e-13:
                nc += 1
                fc += mc / 2 > q2 - q1 + 1e-9
out["C"] = dict(members=nc, fails=int(fc))
print("C", out["C"], flush=True)
rowsD, fd = [], 0
for trial in range(3 if a.smoke else 20):
    K = int(rng.choice([4, 6, 8]))
    k = K // 2
    perm = rng.permutation(K)
    pair = np.zeros(K, dtype=int)
    for j in range(0, K, 2):
        pair[perm[j]], pair[perm[j + 1]] = perm[j + 1], perm[j]
    P = np.zeros((K, K))
    P[np.arange(K), pair] = 1
    Ps = .5 * (np.eye(K) + P) - np.ones((K, K)) / K
    pi = rng.dirichlet(np.ones(K) * 30.)
    t = rng.normal(size=K) * .2
    tau = rng.uniform(.5, 2)
    L = np.tile(t, (K, 1)) + tau * (np.eye(K) - P)
    q = softmax(L)
    mus = []
    for c in range(K):
        mask = np.ones(K, bool)
        mask[pair[c]] = False
        mus.append(min(2 * q[c, mask].min(), .9 * tau / (1 / q[c, mask].min() + 1 / q[c, pair[c]])) * pi[c])
    mu = .5 * min(mus)
    mc = mu / pi
    s = q + mc[:, None] * Ps
    delta = rng.normal(size=(K, K))
    delta -= delta.mean(1, keepdims=True)
    allow = min((q[c, np.arange(K) != pair[c]].min() - mc[c] / 2 for c in range(K)))
    allow = min(allow, q[np.arange(K), pair].min())
    e_in = rng.choice([1e-3, 1e-2, .1]) * allow
    delta = delta * (e_in / np.abs(delta).max())
    st = s + delta
    e = np.abs(delta).sum(1)
    h = lambda x: 0. if x <= 0 else -x * np.log(x) - (1 - x) * np.log(1 - x)
    F = np.array([x / 2 * np.log(K - 1) + h(x / 2) for x in e])
    osc = np.log(q.max(1) / q.min(1))
    bound = float((pi * (.5 * e * osc + 3 * F)).sum())
    loss = float((pi * (st * (np.log(st) - np.log(q))).sum(1)).sum())
    best = fit(st, pi, np.eye(K), k, STARTS)
    gap = loss - best
    rowsD.append(dict(K=K, e=float(e.mean()), bound=bound, gap=gap))
    fd += gap > bound + 1e-9
out["D"] = dict(rows=rowsD, fails=int(fd))
print("D fails", fd, "of", len(rowsD), "largest gap / bound", max(r["gap"] / r["bound"] for r in rowsD), flush=True)
if not a.smoke:
    json.dump(out, open(os.path.join(HERE, "tq18_defect.json"), "w"), indent=1, default=float)
print("done", flush=True)
