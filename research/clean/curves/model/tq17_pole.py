# serves: TQ17
"""Checks of Theorem TQ17. Estimators fixed before the run. float64.
A. The row identity (R) and the entry facts, on random projectors of rank k - 1 orthogonal to 1, K = 4, 6, 8, 12, with random pairings. Fails if |sum E^2 - T| > 1e-10, or an entry fact fails.
B. Facts about phi: phi(y)/y^2 non-increasing, q phi(-w/q) decreasing in q, on grids. Sympy: the row identity and the symmetric double zero of g at T = 1/2.
C. The symmetric table, K = 4, 6, 8, 10, 12, a in {0.3, 0.6, 0.9, 0.99}. Premises of DQ13, by direct evaluation of g on a fine grid plus the proved shape. Value by formula against the build's KL (fails if the difference > 1e-12). 20 random starts of the three-block fit, L-BFGS.
   Prediction, written before the run: no start falls below the formula by more than 1e-9.
D. The pole family, not symmetric: 60 random (pi, t, tau, mu) at K = 4, 6, 8, kept when the premises pass and the table is positive. Value by formula. 20 random starts. Prediction: no start below the value by more than 1e-9, on any instance.
   Also the strict test: the fit's span against the pole span when g > 0 on (0, 1] (largest principal angle at the best start, reported, not a pass/fail).
E. Control: instances that FAIL premise 2 (mu raised until g goes below 0). The theorem says nothing there. Reported: how far a fit falls below the pole value. Prediction: below it, on most of them.
Usage: tq17_pole.py [--smoke] [--threads N]"""
import argparse
import json
import os
import time

import numpy as np
import sympy as sp
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_default_dtype(torch.float64)
torch.set_num_threads(a.threads)
torch.manual_seed(17)
rng = np.random.default_rng(17)
STARTS = 6 if a.smoke else 20


def phi(y):
    y = np.asarray(y, dtype=float)
    return np.where(y <= -1, 1.0, (1 + y) * np.log1p(np.maximum(y, -1 + 1e-300)) - y)


def eta(x):
    return -x * np.log(x)


def pole_projector(K, pair):
    P = np.zeros((K, K))
    for c in range(K):
        P[c, pair[c]] = 1
    return .5 * (np.eye(K) + P) - np.ones((K, K)) / K


def random_pairing(K):
    ix = rng.permutation(K)
    pair = np.zeros(K, dtype=int)
    for j in range(0, K, 2):
        pair[ix[j]] = ix[j + 1]
        pair[ix[j + 1]] = ix[j]
    return pair


def random_projector(K, k):
    X = np.column_stack([np.ones(K), rng.normal(size=(K, k))])
    Q, _ = np.linalg.qr(X)
    return np.eye(K) - Q @ Q.T


def family(pi, t, tau, mu, pair):
    """The table of Definition DQ13, its premises, and its value."""
    K = len(pi)
    Ps = pole_projector(K, pair)
    logits = np.tile(t, (K, 1))
    for c in range(K):
        logits[c, c] += tau
        logits[c, pair[c]] -= tau
    q = np.exp(logits)
    q /= q.sum(1, keepdims=True)
    muc = mu / pi
    s = q + muc[:, None] * Ps
    ok = s.min() > 0
    Ts = np.linspace(1e-6, 1, 4001)
    worst = np.inf
    strict = True
    for c in range(K):
        mask = np.ones(K, bool)
        mask[pair[c]] = False
        m = q[c, mask].min()
        if m <= muc[c] / 2:
            ok = False
            continue
        gam = 4 * m * float(phi(-muc[c] / (2 * m)))
        qp = q[c, pair[c]]
        g = (muc[c] * tau - gam) * Ts + gam * Ts ** 2 - qp * phi(muc[c] * Ts / qp)
        worst = min(worst, float(g.min()))
        strict = strict and bool((g > 0).all())
    ok = ok and worst >= -1e-14
    val = float(sum(pi[c] * (eta(q[c]).sum() - eta(s[c]).sum()) for c in range(K)))
    return dict(s=s, q=q, ok=bool(ok), worst_g=worst, strict=strict, value=val)


def kl_tab(s, l, pi):
    return (pi * (s * (s.log() - torch.log_softmax(l, 1))).sum(1)).sum()


def fit_table(s, pi, k, starts):
    s = torch.tensor(s)
    pi = torch.tensor(pi)
    K, C = s.shape
    best, vals, bestV = 1e9, [], None
    for j in range(starts):
        V = (torch.randn(C, k) * .8).requires_grad_(True)
        c = (torch.randn(K, k) * .8).requires_grad_(True)
        t = torch.zeros(C).requires_grad_(True)
        opt = torch.optim.LBFGS([V, c, t], lr=1, max_iter=3000, history_size=60, line_search_fn="strong_wolfe", tolerance_grad=1e-11, tolerance_change=1e-16)

        def cl():
            opt.zero_grad()
            v = kl_tab(s, t + c @ V.T, pi)
            v.backward()
            return v
        opt.step(cl)
        v = float(kl_tab(s, t + c @ V.T, pi).detach())
        vals.append(v)
        if v < best:
            best, bestV = v, V.detach().numpy().copy()
    return best, float(np.median(vals)), bestV


def span_angle(V, pair):
    K, k = V.shape
    A = np.column_stack([np.ones(K), V])
    Q1, _ = np.linalg.qr(A)
    cols = [np.ones(K)]
    seen = set()
    for c in range(K):
        if c not in seen:
            e = np.zeros(K)
            e[c] = 1
            e[pair[c]] = -1
            cols.append(e)
            seen |= {c, int(pair[c])}
    Q2, _ = np.linalg.qr(np.column_stack(cols))
    sv = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float(np.arccos(np.clip(sv.min(), -1, 1)))


out = {}
t0 = time.time()
# A
fails, n = 0, 0
for K in (4, 6, 8, 12):
    k = K // 2
    beta, tt = .5 - 1 / K, -1 / K
    for _ in range(200 if a.smoke else 5000):
        P = random_projector(K, k)
        pair = random_pairing(K)
        Ps = pole_projector(K, pair)
        for c in range(K):
            cp = pair[c]
            E = P[c] - Ps[c]
            T = beta - P[c, cp]
            n += 1
            if abs((E ** 2).sum() - T) > 1e-10:
                fails += 1
            off = np.delete(P[c], c)
            if off.max() > beta + 1e-12 or P[c, c] > 1 - 1 / K + 1e-12 or T < -1e-12 or T > 1 + 1e-12:
                fails += 1
out["A"] = dict(rows=n, fails=fails)
print("A", out["A"], flush=True)
# B
ys = np.linspace(-.999, 40, 40000)
yn = ys[np.abs(ys) > 1e-3]
r = phi(yn) / yn ** 2
mono = bool((np.diff(r) <= 1e-9).all())
w = .3
qs = np.linspace(w + 1e-4, 5, 20000)
dec = bool((np.diff(qs * phi(-w / qs)) <= 1e-12).all())
K_, d_, p_ = sp.symbols("K d p", positive=True)
beta_ = sp.Rational(1, 2) - 1 / K_
A_, B_ = d_ - d_ ** 2 - p_ ** 2, -d_ - p_
lhs = (d_ - beta_) ** 2 + (p_ - beta_) ** 2 + A_ + 2 * B_ / K_ + (K_ - 2) / K_ ** 2
sym_id = sp.simplify(lhs - (beta_ - p_))
out["B"] = dict(ratio_monotone=mono, q_phi_decreasing=dec, sympy_row_identity_residual=str(sym_id))
print("B", out["B"], flush=True)
# C
rowsC = []
Fc = 0
for K in ((4, 6) if a.smoke else (4, 6, 8, 10, 12)):
    k = K // 2
    for ac in (.3, .6, .9, .99):
        b = (1 - ac) / (K - 1)
        r_ = ac - b
        u = (r_ * (K - 2) + np.sqrt(r_ ** 2 * (K - 2) ** 2 + 4 * (1 - r_) * (1 + r_))) / (2 * (1 - r_))
        Z = u + 1 / u + K - 2
        pair = random_pairing(K)
        pi = np.ones(K) / K
        mu = (2 * (u - 1) / (u * Z)) / K
        fam = family(pi, np.zeros(K), np.log(u), mu, pair)
        s0 = np.full((K, K), b) + np.eye(K) * r_
        formula = np.log(Z) - r_ * np.log(u) + (eta(np.array([ac])).sum() + (K - 1) * eta(np.array([b])).sum()) * -1
        tb = float(np.abs(fam["s"] - s0).max())
        best, med, V = fit_table(s0, pi, k, STARTS)
        row = dict(K=K, a=ac, premises=fam["ok"], worst_g=fam["worst_g"], table_err=tb, formula=float(formula), value_from_family=fam["value"], least_found=best, median=med, below=float(formula) - best)
        if not fam["ok"] or tb > 1e-12 or abs(row["formula"] - fam["value"]) > 1e-12 or row["below"] > 1e-9:
            Fc += 1
        rowsC.append(row)
        print(row, flush=True)
out["C"] = dict(rows=rowsC, fails=Fc)
print("C fails", Fc, flush=True)
# D
rowsD = []
Fd = 0
tries = 0
target = 6 if a.smoke else 60
while len(rowsD) < target and tries < 100000:
    tries += 1
    K = int(rng.choice([4, 6, 8]))
    k = K // 2
    pi = rng.dirichlet(np.ones(K) * rng.choice([3., 30., 300.]))
    t = rng.normal(size=K) * rng.choice([.1, .4, 1.])
    tau = float(rng.uniform(.3, 3))
    pair = random_pairing(K)
    # mu scaled to the largest size the premises allow, on a grid, then a random fraction of it, so that instances are not all near the fit
    probe = family(pi, t, tau, 1e-3, pair)
    if not probe["ok"]:
        continue
    lo = 1e-3
    for f in np.geomspace(1e-3, 1, 25):
        if family(pi, t, tau, float(f), pair)["ok"]:
            lo = float(f)
    mu = lo * float(rng.uniform(.3, 1))
    fam = family(pi, t, tau, mu, pair)
    if not fam["ok"]:
        continue
    best, med, V = fit_table(fam["s"], pi, k, STARTS)
    ang = span_angle(V, pair)
    row = dict(K=K, tau=tau, mu=mu, pi_spread=float(pi.max() / pi.min()), t_spread=float(t.max() - t.min()), strict=fam["strict"], value=fam["value"], least_found=best, median=med, below=fam["value"] - best, angle_best_to_pole=ang)
    if row["below"] > 1e-9:
        Fd += 1
    rowsD.append(row)
    print(row, flush=True)
    if not a.smoke:
        out["D"] = dict(rows=rowsD, fails=Fd)
        json.dump(out, open(os.path.join(HERE, "tq17_pole.json"), "w"), indent=1)
out["D"] = dict(rows=rowsD, fails=Fd)
print("D fails", Fd, "of", len(rowsD), flush=True)
# E
rowsE = []
trial = 0
while len(rowsE) < (3 if a.smoke else 15) and trial < 5000:
    trial += 1
    K = int(rng.choice([4, 6, 8]))
    k = K // 2
    pi = np.ones(K) / K
    t = rng.normal(size=K) * .1
    tau = float(rng.uniform(.3, 1.5))
    pair = random_pairing(K)
    good = None
    for f in np.geomspace(1e-3, 1, 40):
        fam = family(pi, t, tau, float(f), pair)
        if fam["s"].min() > 0 and not fam["ok"]:
            good = (float(f), fam)
            break
    if good is None:
        continue
    f, fam = good
    best, med, V = fit_table(fam["s"], pi, k, STARTS)
    row = dict(K=K, tau=tau, mu=f, worst_g=fam["worst_g"], pole_value=fam["value"], least_found=best, below=fam["value"] - best)
    rowsE.append(row)
    print(row, flush=True)
out["E"] = rowsE
out["sec"] = time.time() - t0
if not a.smoke:
    json.dump(out, open(os.path.join(HERE, "tq17_pole.json"), "w"), indent=1)
print("done", out["sec"], flush=True)
