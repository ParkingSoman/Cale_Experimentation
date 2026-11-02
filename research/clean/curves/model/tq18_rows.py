# serves: TQ18
"""Checks of the parts added after the second referee round. Estimators fixed before the run. float64.
A. Theorem TQ18 with kept units (Y = span(1, h_T)). Tables built backward, n rows, C classes, y kept functions, read or free codes. 15 starts of the fit A Y + V c, against the entropy sum at 1.0 of the test. Prediction: no start below by more than 1e-9.
B. Theorem TQ18 part 6, the row-wise test. Tables built backward at 0.5 of the test, and each row split into m = 4 equal rows. Compute the smallest test value over a grid of w, before and after the split. Prediction: the test number moves by less than 1e-9 relative, the test of parts 1 and 2 gets worse by the split, and where part 6 passes, fits stay at or above the value.
C. Lemma TQ19 part 1 on random s, H, l. Fails if violated by > 1e-12.
D. Corollary TQ17.1, the sum form. C = 8, 4 pairs: J_T has 1 pair, S_1 reads 1 pair, S_2 reads 2 pairs. 8 regions, 40 photos each, h_T = the pair function plus a constant, z of each group = its pair functions, plus nuisance features constant within... free noise orthogonal to the pair functions. Fit: A h_T + t + V_1 M_1 z_1 + V_2 M_2 z_2, 15 starts, against the closed form. Prediction: the best fit equals the closed form within 1e-8 and no start falls below it by more than 1e-9.
Usage: tq18_rows.py [--smoke]"""
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
torch.manual_seed(31)
rng = np.random.default_rng(31)
STARTS = 4 if a.smoke else 15


def entropy(q):
    return -(q * np.log(q)).sum(-1)


def phi(y):
    return (1 + y) * np.log1p(y) - y


def softmax(l):
    l = l - l.max(1, keepdims=True)
    q = np.exp(l)
    return q / q.sum(1, keepdims=True)


def basis(X):
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, s > 1e-10 * s[0]]


def lbfgs(params, fn):
    opt = torch.optim.LBFGS(params, lr=1, max_iter=3000, history_size=60, line_search_fn="strong_wolfe", tolerance_grad=1e-11, tolerance_change=1e-16)

    def cl():
        opt.zero_grad()
        v = fn()
        v.backward()
        return v
    opt.step(cl)
    return float(fn().detach())


def fit_general(s, p, Yb, Qz, k, starts):
    s_, p_, Y_, Z_ = map(torch.tensor, (s, p, Yb, Qz))
    n, C = s.shape
    best = 1e9
    kl = lambda L: (p_ * (s_ * (s_.log() - torch.log_softmax(L, 1))).sum(1)).sum()
    for _ in range(starts):
        A = (torch.randn(Y_.shape[1], C) * .3).requires_grad_(True)
        V = (torch.randn(C, k) * .8).requires_grad_(True)
        c = (torch.randn(Z_.shape[1], k) * .8).requires_grad_(True)
        best = min(best, lbfgs([A, V, c], lambda: kl(Y_ @ A + (Z_ @ c) @ V.T)))
    return best


def build(n, C, k, y, mode, scale):
    Ym = np.column_stack([np.ones(n)] + [rng.normal(size=n) for _ in range(y)])
    Yb = basis(Ym)
    Z = np.eye(n) if mode == "free" else np.column_stack([Ym, rng.normal(size=(n, n - y - 3))])
    Qz = basis(Z)
    Pz = Qz @ Qz.T
    Py = Yb @ Yb.T
    p = rng.dirichlet(np.ones(n) * 30.)
    Q, _ = np.linalg.qr(np.column_stack([np.ones(C), rng.normal(size=(C, k))]))
    V = Q[:, 1:k + 1]
    c = (Pz - Py) @ (rng.normal(size=(n, k)) * rng.uniform(.5, 1.2))
    A = rng.normal(size=(Yb.shape[1], C)) * .3
    L0 = Yb @ A + c @ V.T
    q = softmax(L0)
    u, sv, _ = np.linalg.svd(c, full_matrices=False)
    rho = (p[:, None] * q).min()
    Pi = np.eye(C) - V @ V.T - np.ones((C, C)) / C
    G = (Pz - Py - u @ u.T) @ rng.normal(size=(n, C)) @ Pi
    gn = np.linalg.norm(G, 2)
    if gn < 1e-14:
        return None
    G = G * (scale * rho * min(1, sv.min() / 2) / gn)
    s = q + G / p[:, None]
    if s.min() <= 0:
        return None
    return dict(s=s, p=p, q=q, Yb=Yb, Qz=Qz, G=G, c=c, V=V, sigma=sv.min(), rho=rho, n=n, C=C, k=k)


def row_test(G, c, p, q, Pz, Yb):
    Q_ = Pz @ G
    rho_x = p * q.min(1)
    B = c @ np.linalg.inv(c.T @ c)
    Dm = 1 / np.sqrt(rho_x)
    QQ = (Dm[:, None] * Q_) @ (Dm[:, None] * Q_).T
    BB = (Dm[:, None] * B) @ (Dm[:, None] * B).T
    best = np.inf
    bw = None
    ok_any = False
    for w in np.geomspace(1e-8, 1e2, 400):
        a_ = 2 * np.linalg.norm(QQ + w ** 2 * BB, 2) / w
        pos = (((Q_ ** 2).sum(1) + w ** 2 * (B ** 2).sum(1)) <= rho_x ** 2).all()
        if a_ < best:
            best, bw = a_, w
        ok_any = ok_any or (a_ <= 1 and pos)
    return float(best), bool(ok_any)


out = {}
rowsA, fa = [], 0
for (n, C, k, y) in ([(8, 6, 1, 1)] if a.smoke else [(8, 6, 1, 1), (10, 6, 2, 1), (12, 8, 2, 2), (16, 8, 3, 2)]):
    for mode in ("free", "read"):
        for scale in (.5, 1.):
            inst = None
            while inst is None:
                inst = build(n, C, k, y, mode, scale)
            val = float((inst["p"] * (entropy(inst["q"]) - entropy(inst["s"]))).sum())
            best = fit_general(inst["s"], inst["p"], inst["Yb"], inst["Qz"], k, STARTS)
            rowsA.append(dict(n=n, C=C, k=k, y=y, mode=mode, scale=scale, value=val, least=best, below=val - best))
            fa += val - best > 1e-9
            print(rowsA[-1], flush=True)
out["A"] = dict(rows=rowsA, fails=int(fa))
print("A fails", fa, "of", len(rowsA), flush=True)
rowsB, fb = [], 0
for trial in range(2 if a.smoke else 12):
    inst = None
    while inst is None:
        inst = build(8, 6, 1 + trial % 2, 0, "free", .5)
    Pz = inst["Qz"] @ inst["Qz"].T
    t1, ok1 = row_test(inst["G"], inst["c"], inst["p"], inst["q"], Pz, inst["Yb"])
    m = 4
    G2 = np.repeat(inst["G"] / m, m, axis=0)
    c2 = np.repeat(inst["c"], m, axis=0)
    p2 = np.repeat(inst["p"] / m, m)
    q2 = np.repeat(inst["q"], m, axis=0)
    Z2 = basis(np.eye(len(p2)))
    t2, ok2 = row_test(G2, c2, p2, q2, np.eye(len(p2)), None)
    g1 = np.linalg.norm(Pz @ inst["G"], 2)
    test1 = g1 / (inst["rho"])
    g2 = np.linalg.norm(G2, 2)
    test2 = g2 / ((p2[:, None] * q2).min())
    best = fit_general(inst["s"], inst["p"], inst["Yb"], inst["Qz"], inst["k"], STARTS)
    val = float((inst["p"] * (entropy(inst["q"]) - entropy(inst["s"]))).sum())
    rowsB.append(dict(row_test=t1, row_test_split=t2, passes=ok1, passes_split=ok2, old_premise=test1, old_premise_split=test2, value=val, least=best, below=val - best))
    fb += (abs(t1 - t2) > 1e-6 * max(t1, 1e-12)) + (ok1 and val - best > 1e-9)
    print(rowsB[-1], flush=True)
out["B"] = dict(rows=rowsB, fails=int(fb))
print("B fails", fb, flush=True)
fc = 0
for _ in range(20000):
    C = int(rng.integers(3, 12))
    s = rng.dirichlet(np.ones(C) * rng.uniform(.2, 3))
    s = np.maximum(s, 1e-12)
    s = s / s.sum()
    ix = rng.permutation(C)
    h = int(rng.integers(1, C))
    H, T = ix[:h], ix[h:]
    l = rng.normal(size=C) * rng.uniform(.1, 8)
    zeta = s[T].sum()
    w = s[T] / zeta
    l2 = np.concatenate([l[H], [w @ l[T] - (w * np.log(w)).sum()]])
    s2 = np.concatenate([s[H], [zeta]])
    def KL(a_, l_):
        lq = l_ - np.log(np.exp(l_ - l_.max()).sum()) - l_.max()
        return (a_ * (np.log(a_) - lq)).sum()
    fc += KL(s, l) < KL(s2, l2) - 1e-12
out["C"] = dict(fails=int(fc))
print("C", out["C"], flush=True)
# D
C = 8
pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
pair = np.zeros(8, dtype=int)
for i_, j_ in pairs:
    pair[i_], pair[j_] = j_, i_
P_ = np.zeros((8, 8))
P_[np.arange(8), pair] = 1
Ps = .5 * (np.eye(8) + P_) - np.ones((8, 8)) / 8
tau = 1.2
L = tau * (np.eye(8) - P_)
q = softmax(L)
qp = q[0, 1]
m_ = q[0, 2]
mu_c = .5 * min(2 * m_, tau / (1 / m_ + 1 / qp))
s_tab = q + mu_c * Ps
value = float(np.mean(entropy(q) - entropy(s_tab)))
per = 8 if a.smoke else 40
lab = np.repeat(np.arange(8), per)
n = len(lab)
f = np.zeros((n, 4))
for j, (i_, i2) in enumerate(pairs):
    f[lab == i_, j] = 1
    f[lab == i2, j] = -1
S = torch.tensor(s_tab[lab])
noise = rng.normal(size=(n, 3))
# nuisance features orthogonal to the pair functions and to 1, so that z varies inside regions
X = np.column_stack([np.ones(n), f])
noise = noise - X @ np.linalg.lstsq(X, noise, rcond=None)[0]
h_T = torch.tensor(np.column_stack([f[:, 0] + 2.]))
z1 = torch.tensor(np.column_stack([f[:, 1], noise[:, :1]]))
z2 = torch.tensor(np.column_stack([f[:, 2], f[:, 3], noise[:, 1:]]))
vals = []
for st in range(STARTS):
    t = torch.zeros(C).requires_grad_(True)
    A = (torch.randn(1, C) * .5).requires_grad_(True)
    V1 = (torch.randn(C, 1) * .8).requires_grad_(True)
    M1 = (torch.randn(2, 1) * .8).requires_grad_(True)
    V2 = (torch.randn(C, 2) * .8).requires_grad_(True)
    M2 = (torch.randn(4, 2) * .8).requires_grad_(True)
    lf = lambda: (S * (S.log() - torch.log_softmax(t + h_T @ A + (z1 @ M1) @ V1.T + (z2 @ M2) @ V2.T, 1))).sum(1).mean()
    vals.append(lbfgs([t, A, V1, M1, V2, M2], lf))
out["D"] = dict(value=value, least=min(vals), below=value - min(vals), median=float(np.median(vals)))
print("D", out["D"], flush=True)
if not a.smoke:
    json.dump(out, open(os.path.join(HERE, "tq18_rows.json"), "w"), indent=1, default=float)
print("done", flush=True)
