# serves: TQ5
"""Check of Theorem TQ5 and Lemma TQ6 (COARSE-THEORY.md B.Q), at the logit level of the setting of B.Q.
Instance: n areas, each with a block K_w of classes, plus classes in no block; on A_w the centred logits of N are
P_perp y = ybar_w + B_w ztil_w(x) (linear-read claim at eps = 0), ztil_w centred on A_w; own-class rank of each area
<= k (squeeze-complete); N_m lowers every class outside K_w by m on A_w.
A (part 2, no optimiser): random EXACTLY feasible leak points (L, g, beta); the box of the proof (logits
   L_w ztil + t_w with t_w as in the proof) must have KL terms <= e^-m Lambda(L, g, beta) for every m (to 1e-12 rel).
B (part 3): Lambda_S by a penalty method over leak points (restarts) and L^on_S(m) by direct minimisation over
   always-on stacked maps of rank k (restarts), m = 1..10: e^m L^on_S(m) <= Lambda_S (upper side, to 1e-6) and the
   ratio e^m L^on_S(m) / Lambda_S -> 1. Local optimisers: B reports, A and C certify.
C (Lemma TQ6): random rank-k matrices with unit diagonal stay >= n^2/k - n; the harmonic projection reaches it.
Run with /opt/homebrew/bin/python3.11 (torch)."""
import json
import math
import os

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20261101)
torch.manual_seed(20261101)
fails = []


def make_instance(n_areas=3, blk=2, rest=1, d=2, pts=6, k=2, own_rank=1, scale=1.0):
    ny = n_areas * blk + rest
    K = [list(range(w * blk, (w + 1) * blk)) for w in range(n_areas)]
    areas = []
    for w in range(n_areas):
        Z = rng.normal(size=(pts, d)); Z -= Z.mean(0)            # centred (equal weights)
        p = np.full(pts, 1.0 / (pts * n_areas))
        B = rng.normal(size=(ny, d)) * 0.4 * scale             # outside movement of N
        U = rng.normal(size=(blk, own_rank)); Vb = rng.normal(size=(own_rank, d))
        B[K[w]] = scale * U @ Vb                                # own block, rank own_rank
        ybar = rng.normal(size=ny) * 0.5
        areas.append(dict(Z=Z, p=p, B=B, K=K[w], ybar=ybar))
    return dict(ny=ny, K=K, areas=areas, k=k)


def Pperp(ny):
    return np.eye(ny) - np.ones((ny, ny)) / ny


def logits_Nm(inst, w, m):
    a = inst["areas"][w]; ny = inst["ny"]
    Y = (Pperp(ny) @ a["B"] @ a["Z"].T).T                       # Y_w(x), P_perp-valued
    out = np.ones(ny); out[a["K"]] = 0
    return a["ybar"][None, :] + Y - m * out[None, :]


def kl_rows(y, l):
    ls = y - np.logaddexp.reduce(y, axis=1, keepdims=True)
    lq = l - np.logaddexp.reduce(l, axis=1, keepdims=True)
    return (np.exp(ls) * (ls - lq)).sum(1)


def rho(inst, w):
    a = inst["areas"][w]; y0 = logits_Nm(inst, w, 0.0)
    lse = np.logaddexp.reduce(y0[:, a["K"]], axis=1, keepdims=True)
    r = np.exp(y0 - lse); r[:, a["K"]] = 0
    return r


def phi(t):
    return np.expm1(t) - t


def lam_value(inst, L, g, beta):
    tot = 0.0
    for w, a in enumerate(inst["areas"]):
        Y = (Pperp(inst["ny"]) @ a["B"] @ a["Z"].T).T
        dd = a["Z"] @ L[w].T - Y - (a["Z"] @ g[w])[:, None] - beta[w][None, :]
        r = rho(inst, w)
        tot += float((a["p"][:, None] * r * phi(dd)).sum())
    return tot


def random_leak_point(inst):
    ny, k = inst["ny"], inst["k"]; P = Pperp(ny)
    V = rng.normal(size=(ny, k)); L = []; g = []; beta = []
    Phis = [rng.normal(size=(k, a["Z"].shape[1])) for a in inst["areas"]]
    for w, a in enumerate(inst["areas"]):          # V rows on K_w: V_K Phi_w = Pi^K B_w + 1 g^T (Phi_w injective)
        gw = rng.normal(size=a["Z"].shape[1])
        target = a["B"][a["K"]] + np.outer(np.ones(len(a["K"])), gw)
        V[a["K"]] = target @ np.linalg.pinv(Phis[w])
    for w, a in enumerate(inst["areas"]):
        Lw = P @ V @ Phis[w]
        Yk = (P @ a["B"])[a["K"]]
        diff = Lw[a["K"]] - Yk
        gw = diff.mean(0)
        if np.abs(diff - gw[None, :]).max() > 1e-9:
            fails.append(("A-feasible", float(np.abs(diff - gw).max())))
        L.append(Lw); g.append(gw); beta.append(rng.normal(size=ny))
    return L, g, beta


# ---------- A ----------
nA = 0
for trial in range(40):
    inst = make_instance(n_areas=int(rng.integers(2, 5)), blk=int(rng.integers(2, 4)), rest=int(rng.integers(0, 3)),
                         d=int(rng.integers(1, 3)), k=int(rng.integers(2, 4)), own_rank=1, scale=float(rng.uniform(.3, 2)))
    L, g, beta = random_leak_point(inst)
    lam = lam_value(inst, L, g, beta)
    for m in [0.0, 0.5, 2.0, 5.0, 10.0]:
        tot = 0.0
        for w, a in enumerate(inst["areas"]):
            t = a["ybar"].copy(); out = [i for i in range(inst["ny"]) if i not in a["K"]]
            t[out] = a["ybar"][out] - m - beta[w][out]
            l = a["Z"] @ L[w].T + t[None, :]
            tot += float((a["p"] * kl_rows(logits_Nm(inst, w, m), l)).sum())
        nA += 1
        if tot > math.exp(-m) * lam * (1 + 1e-12) + 1e-15:
            fails.append(("A-upper", trial, m, tot, math.exp(-m) * lam))

# ---------- B ----------
def solve_lambda(inst, restarts=6, iters=4):
    ny, k = inst["ny"], inst["k"]; P = torch.tensor(Pperp(ny))
    best = np.inf
    data = []
    for w, a in enumerate(inst["areas"]):
        data.append((torch.tensor(a["Z"]), torch.tensor(a["p"]), torch.tensor((Pperp(ny) @ a["B"] @ a["Z"].T).T),
                     torch.tensor(rho(inst, w)), a["K"]))
    for _ in range(restarts):
        V = torch.randn(ny, k, requires_grad=True)
        Ph = [torch.randn(k, a["Z"].shape[1], requires_grad=True) for a in inst["areas"]]
        G = [torch.zeros(a["Z"].shape[1], requires_grad=True) for a in inst["areas"]]
        Bt = [torch.zeros(ny, requires_grad=True) for _ in inst["areas"]]
        params = [V] + Ph + G + Bt
        for pen in [1e1, 1e3, 1e5, 1e7][:iters]:
            opt = torch.optim.LBFGS(params, max_iter=3000, line_search_fn="strong_wolfe", tolerance_grad=1e-13,
                                    tolerance_change=1e-16)
            def f():
                opt.zero_grad(); tot = 0.0; viol = 0.0
                for w, (Z, p, Y, r, Kw) in enumerate(data):
                    Lz = Z @ (P @ V @ Ph[w]).T
                    gz = Z @ G[w]
                    viol = viol + ((Lz[:, Kw] - Y[:, Kw] - gz[:, None]) ** 2).sum()
                    dd = Lz - Y - gz[:, None] - Bt[w][None, :]
                    mask = torch.ones(ny); mask[Kw] = 0
                    tot = tot + (p[:, None] * r * (torch.expm1(dd) - dd) * mask).sum()
                v = tot + pen * viol; v.backward(); return v
            opt.step(f)
        with torch.no_grad():
            tot = 0.0; viol = 0.0
            for w, (Z, p, Y, r, Kw) in enumerate(data):
                Lz = Z @ (P @ V @ Ph[w]).T; gz = Z @ G[w]
                viol = max(viol, float((Lz[:, Kw] - Y[:, Kw] - gz[:, None]).abs().max()))
                dd = Lz - Y - gz[:, None] - Bt[w][None, :]
                mask = torch.ones(ny); mask[Kw] = 0
                tot = tot + float((p[:, None] * r * (torch.expm1(dd) - dd) * mask).sum())
        if viol < 1e-6 and tot < best:
            best = tot
    return best


def solve_Lon(inst, m, restarts=6):
    ny, k = inst["ny"], inst["k"]; best = np.inf
    data = [(torch.tensor(a["Z"]), torch.tensor(a["p"]), torch.tensor(logits_Nm(inst, w, m)))
            for w, a in enumerate(inst["areas"])]
    for _ in range(restarts):
        V = torch.randn(ny, k, requires_grad=True)
        Ph = [torch.randn(k, a["Z"].shape[1], requires_grad=True) for a in inst["areas"]]
        T = [torch.tensor(a["ybar"] - m * (1 - np.isin(np.arange(ny), a["K"])), requires_grad=True)
             for a in inst["areas"]]
        params = [V] + Ph + T
        opt = torch.optim.LBFGS(params, max_iter=5000, line_search_fn="strong_wolfe", tolerance_grad=1e-15,
                                tolerance_change=1e-18)
        def f():
            opt.zero_grad(); tot = 0.0
            for w, (Z, p, y) in enumerate(data):
                l = Z @ (V @ Ph[w]).T + T[w][None, :]
                ls = y - torch.logsumexp(y, 1, keepdim=True); lq = l - torch.logsumexp(l, 1, keepdim=True)
                tot = tot + (p * (torch.exp(ls) * (ls - lq)).sum(1)).sum()
            v = tot * math.exp(m); v.backward(); return v
        for _ in range(3):
            opt.step(f)
        v = float(f()) * math.exp(-m)
        best = min(best, v)
    return best


rowsB = []
for trial in range(4):
    inst = make_instance(n_areas=3, blk=2, rest=1, d=1, pts=5, k=2, own_rank=1, scale=1.0)
    lam = solve_lambda(inst)
    row = dict(trial=trial, Lambda=lam, ratios=[])
    for m in [1.0, 3.0, 5.0, 7.0, 9.0]:
        lon = solve_Lon(inst, m)
        ratio = math.exp(m) * lon / lam
        row["ratios"].append((m, ratio))
        if ratio > 1 + 1e-6:
            fails.append(("B-upper", trial, m, ratio))
    rowsB.append(row)
    print("B", trial, "Lambda %.6g" % lam, " e^m L_on / Lambda:", ["%.5f" % r for _, r in row["ratios"]], flush=True)

# ---------- C ----------
nC = 0
for trial in range(300):
    n = int(rng.integers(2, 9)); k = int(rng.integers(1, n + 1))
    R = rng.normal(size=(n, k)); E = rng.normal(size=(k, n)); X = R @ E
    dg = np.diag(X).copy()
    if np.min(np.abs(dg)) < 1e-3:
        continue
    X = X / dg[:, None]
    off = float((X ** 2).sum() - n)
    nC += 1
    if off < n * n / k - n - 1e-9:
        fails.append(("C-bound", n, k, off))
for n in range(2, 12):
    for k in range(1, n + 1):
        if n == k:
            P = np.eye(n)
        else:
            rows = []
            for l in range(1, k // 2 + 1):
                j = np.arange(1, n + 1)
                rows += [np.sqrt(2 / n) * np.cos(2 * np.pi * l * j / n), np.sqrt(2 / n) * np.sin(2 * np.pi * l * j / n)]
            if k % 2:
                rows.append(np.ones(n) / np.sqrt(n))
            F = np.array(rows); P = F.T @ F
        X = (n / k) * P
        if np.abs(np.diag(X) - 1).max() > 1e-12 or abs((X ** 2).sum() - n - (n * n / k - n)) > 1e-9 \
                or np.linalg.matrix_rank(X) != k:
            fails.append(("C-harmonic", n, k))
        nC += 1
print("A checks", nA, " C checks", nC, " fails", len(fails), fails[:5])
json.dump(dict(fails=fails, A=nA, C=nC, B=rowsB), open(os.path.join(HERE, "tq_leak.json"), "w"), indent=1)
