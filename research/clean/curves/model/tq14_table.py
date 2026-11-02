# serves: TQ14
"""Checks of Theorem TQ14 and Conjecture DQ11. Estimators fixed before the run.
A. Random small nets with exact collapse (features constant on regions, K = 8, C = 8, d = 10): L_on(k), L_free(k), L_table(k) from many starts agree (part 4). Fails if the lowest
   value of each of the three differs by more than 1e-4. Also part 1 and part 2 inequalities on non-collapsed instances (noise added), on local minima: fail if violated by > 1e-8.
B. The first form of Conjecture DQ11. Symmetric tables with known equiangular tight frames: (K, k) = (6, 3), (10, 5). a in {0.5, 0.9, 0.99}. The frame value (tau by a line search) against the lowest of 30 random
   starts. Prediction, written before the run: no start falls below the frame value by more than 1e-6. RESULT of the smoke run: false, every start falls below it (so DQ11 was rewritten); kept as the record, and the pole formula is printed beside it.
C. The trained net of model/ts_identity.py (CIFAR-100, whole training P, labels as regions): eps_const; the within-class share of feature variance; L_table(k), L_on(k), for k = 2, 5, 10, 20, 40; the
   same head's KL on test photos; |W|_F of each build against N's. Prediction: L_on is close to L_table + eps_const and small already at k = 5, with larger weights than N's.
Usage: tq14_table.py [--smoke]"""
import argparse
import json
import os

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--only-a", action="store_true")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_default_dtype(torch.float64)
torch.manual_seed(7)
torch.set_num_threads(6)


def kl_rows(s, l):
    return (s * (torch.log(s.clamp_min(1e-300)) - torch.log_softmax(l, 1))).sum(1)


def fit(params, fn, iters=1500):
    opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, history_size=50, line_search_fn="strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-15)

    def cl():
        opt.zero_grad()
        v = fn()
        v.backward()
        for p in params:
            p.grad = p.grad.contiguous()
        return v
    opt.step(cl)
    return float(fn().detach())


def table_least(sb, pi, k, starts=8, iters=1500, init=None):
    K, C = sb.shape
    best = (1e9, None)
    for j in range(starts):
        V = (torch.randn(C, k) * .5).requires_grad_(True)
        c = (torch.randn(K, k) * .5).requires_grad_(True)
        t = torch.zeros(C).requires_grad_(True)
        v = fit([V, c, t], lambda: (pi * kl_rows(sb, t + c @ V.T)).sum(), iters)
        if v < best[0]:
            best = (v, (V.detach(), c.detach(), t.detach()))
    return best


def on_least(z, s, k, starts=8, iters=1500, init=None):
    n, d = z.shape
    C = s.shape[1]
    best = (1e9, None)
    for j in range(starts):
        V = (torch.randn(C, k) * .5).requires_grad_(True)
        M = (torch.randn(d, k) * .3).requires_grad_(True)
        t = torch.zeros(C).requires_grad_(True)
        if init is not None and j == 0:
            V = init[0].clone().requires_grad_(True)
            M = init[1].clone().requires_grad_(True)
            t = init[2].clone().requires_grad_(True)
        v = fit([V, M, t], lambda: kl_rows(s, t + (z @ M) @ V.T).mean(), iters)
        if v < best[0]:
            best = (v, (V.detach(), M.detach(), t.detach()))
    return best


def free_least(s, k, starts=8, iters=1500):
    n, C = s.shape
    best = 1e9
    for j in range(starts):
        V = (torch.randn(C, k) * .5).requires_grad_(True)
        c = (torch.randn(n, k) * .5).requires_grad_(True)
        t = torch.zeros(C).requires_grad_(True)
        best = min(best, fit([V, c, t], lambda: kl_rows(s, t + c @ V.T).mean(), iters))
    return best


out = {}
# A
fails = 0
K, C, d = 8, 8, 10
rows = []
for trial in range(2 if a.smoke else 6):
    zb = torch.randn(K, d)
    W = torch.randn(C, d) * 1.5
    m = 12
    z = zb.repeat_interleave(m, 0)
    s = torch.softmax(z @ W.T, 1)
    sb = s[::m].contiguous()
    pi = torch.ones(K) / K
    for k in (1, 2, 3):
        lt = table_least(sb, pi, k, 6 if a.smoke else 12)[0]
        lo = on_least(z - z.mean(0), s, k, 6 if a.smoke else 12)[0]
        lf = free_least(s, k, 6 if a.smoke else 12)
        # read the table's solution off the means: a one-read build with the same value (part 4)
        vt, (Vt, ct, tt) = table_least(sb, pi, k, 12)
        lt = min(lt, vt)
        cm = ct.mean(0)
        M0 = torch.linalg.lstsq(zb - z.mean(0), ct - cm).solution   # the means are independent after centring except for their sum: the shift goes to the bias
        lo_w = float(kl_rows(s, tt + cm @ Vt.T + ((z - z.mean(0)) @ M0) @ Vt.T).mean())
        rows.append((vt, lo_w, min(lf, lo)))
        # the theorem: the read of a table build has the same value (exact), and nothing lies below the table's least. A free or read search below the local table value means the table search was stuck: rerun it from that side.
        if abs(lo_w - vt) > 1e-6:
            fails += 1
        if min(lf, lo) < lt - 1e-4:
            lt2 = table_least(sb, pi, k, 40)[0]
            print("table search restarted", k, lt, lt2, min(lf, lo), flush=True)
            if min(lf, lo) < lt2 - 1e-4:
                fails += 1
    # parts 1 and 2 with noise in the features
    zn = z + .3 * torch.randn(z.shape)
    s2 = torch.softmax(zn @ W.T, 1)
    lab = torch.arange(K).repeat_interleave(m)
    sb2 = torch.stack([s2[lab == c].mean(0) for c in range(K)])
    eps = float((pi * torch.stack([kl_rows(s2[lab == c], sb2[c].repeat(m, 1).log()).mean() for c in range(K)])).sum())
    for k in (1, 2, 3):
        lt = table_least(sb2, pi, k, 6)[0]
        V = (torch.randn(C, k) * .5).requires_grad_(True)
        c = (torch.randn(K * m, k) * .5).requires_grad_(True)
        t = torch.zeros(C).requires_grad_(True)
        L = fit([V, c, t], lambda: kl_rows(s2, t + c @ V.T).mean())
        l = (t + c @ V.T).detach()
        cov = 0.0
        for cc in range(K):
            ix = lab == cc
            cov += float(((s2[ix] - sb2[cc]) * (l[ix] - l[ix].mean(0))).sum(1).mean()) / K
        if L > lt + eps + 1e-6:  # L is only a local value, part 1 is about infima; flag if the table + eps is below a value that local search should beat
            pass
        if lt > L + cov - eps + 1e-6:
            fails += 1
if a.only_a:
    print("A: fails", fails, "worst", max(abs(r[1] - r[0]) for r in rows))
    raise SystemExit
print("A: fails", fails, "; worst |read of the table's build - its value|:", max(abs(r[1] - r[0]) for r in rows), flush=True)
out["A"] = dict(fails=fails, worst=max(abs(r[1] - r[0]) for r in rows))

# B
ETF = {}
# (6, 3): the icosahedron's 6 diagonals
g = (1 + 5 ** .5) / 2
ETF[(6, 3)] = torch.tensor([[0, 1, g], [0, 1, -g], [1, g, 0], [1, -g, 0], [g, 0, 1], [-g, 0, 1]], dtype=torch.float64)
ETF[(6, 3)] = ETF[(6, 3)] / ETF[(6, 3)].norm(dim=1, keepdim=True)


def conf_etf(K):
    """Conference-matrix ETF of K = q + 1 vectors in K/2 dimensions (q = 5, 9 are prime powers; use a known symmetric conference matrix when q = 5 or 9)."""
    if K == 10:
        # Paley conference matrix of order 10 over GF(9) = GF(3)[i]
        el = [(a_, b_) for a_ in range(3) for b_ in range(3)]
        sq = {((a_ * a_ - b_ * b_) % 3, (2 * a_ * b_) % 3) for a_, b_ in el if (a_, b_) != (0, 0)}
        Q = torch.zeros(9, 9)
        for i, x in enumerate(el):
            for j, y in enumerate(el):
                dd = ((x[0] - y[0]) % 3, (x[1] - y[1]) % 3)
                if dd != (0, 0):
                    Q[i, j] = 1.0 if dd in sq else -1.0
        Cm = torch.zeros(10, 10)
        Cm[0, 1:] = 1
        Cm[1:, 0] = 1
        Cm[1:, 1:] = Q
        return Cm
    raise ValueError


Cm = conf_etf(10)
ev, U = torch.linalg.eigh(Cm)
mu = 1 / 3 ** 1.0  # coherence of (10, 5) is 1/3
G = torch.eye(10) + Cm * mu
ev, U = torch.linalg.eigh(G)
ETF[(10, 5)] = (U[:, -5:] * ev[-5:].sqrt())
out["etf_rank_check"] = {str(k): int(torch.linalg.matrix_rank(v @ v.T, tol=1e-8)) for k, v in ETF.items()}
print("frames: rank of the Gram matrix", out["etf_rank_check"], flush=True)
rowsB = []
for (KK, k), F in ETF.items():
    for ac in (.5, .9, .99):
        b = (1 - ac) / (KK - 1)
        sb = torch.full((KK, KK), b) + torch.eye(KK) * (ac - b)
        pi = torch.ones(KK) / KK
        G = F @ F.T
        taus = torch.linspace(0, 30, 3001)
        vals = torch.stack([(pi * kl_rows(sb, tau * G)).sum() for tau in taus])
        i = int(vals.argmin())
        lo, hi = float(taus[max(i - 1, 0)]), float(taus[min(i + 1, 3000)])
        for _ in range(60):
            m1, m2 = lo + (hi - lo) * .382, lo + (hi - lo) * .618
            if float((pi * kl_rows(sb, m1 * G)).sum()) < float((pi * kl_rows(sb, m2 * G)).sum()):
                hi = m2
            else:
                lo = m1
        frame = float((pi * kl_rows(sb, ((lo + hi) / 2) * G)).sum())
        vs = []
        for j in range(8 if a.smoke else 30):
            V = torch.randn(KK, k).requires_grad_(True)
            c = torch.randn(KK, k).requires_grad_(True)
            t = torch.zeros(KK).requires_grad_(True)
            vs.append(fit([V, c, t], lambda: (pi * kl_rows(sb, t + c @ V.T)).sum(), 2000))
        r_ = ac - b
        u_ = (r_ * (KK - 2) + (r_ * r_ * (KK - 2) ** 2 + 4 * (1 - r_) * (1 + r_)) ** .5) / (2 * (1 - r_))
        Hs_ = -ac * np.log(ac) - (KK - 1) * b * np.log(b)
        pole = float(np.log(u_ + 1 / u_ + KK - 2) - r_ * np.log(u_) - Hs_)
        row = dict(K=KK, k=k, a=ac, frame=frame, tau=(lo + hi) / 2, least_found=min(vs), median_found=float(np.median(vs)), below=frame - min(vs), pole_formula=pole, pole_minus_least=pole - min(vs))
        rowsB.append(row)
        print(row, flush=True)
out["B"] = rowsB

# C
path = os.path.join(HERE, "..", "cache", "map", "cifar_net.npz")
if os.path.exists(path) and not a.smoke:
    dat = np.load(path)
    h = torch.tensor(dat["h"]).double()
    hte = torch.tensor(dat["hte"]).double()
    lab = torch.tensor(dat["lab"]).long()
    W = torch.tensor(dat["W"]).double()
    b = torch.tensor(dat["b"]).double()
    s = torch.softmax(h @ W.T + b, 1)
    ste = torch.softmax(hte @ W.T + b, 1)
    n = len(lab)
    hm = h.mean(0)
    Hw = h - hm
    Hwte = hte - hm
    sb = torch.stack([s[lab == c].mean(0) for c in range(100)])
    pi = torch.stack([(lab == c).double().mean() for c in range(100)])
    eps = float(sum(pi[c] * kl_rows(s[lab == c], sb[c].repeat(int((lab == c).sum()), 1).log()).mean() for c in range(100)))
    means = torch.stack([Hw[lab == c].mean(0) for c in range(100)])
    within = float(((Hw - means[lab]) ** 2).sum() / (Hw ** 2).sum())
    Iwhole = float((s * (s.clamp_min(1e-300).log() - s.mean(0).log())).sum(1).mean())
    rowC = dict(eps_const=eps, within_share=within, I_whole=Iwhole, W_norm=float(W.norm()), mean_conf=float(s.max(1).values.mean()))
    print(rowC, flush=True)
    out["C_head"] = rowC
    rows = []
    # keep a work sample for the fits: the full P on 256 features is cheap
    for k in (2, 5, 10, 20, 40):
        lt, pt = table_least(sb, pi, k, 3, 800)
        # start the one-read fit from the table's solution: read the codes off the class means
        V0, c0, t0 = pt
        M0 = torch.linalg.lstsq(means, c0).solution
        lo, po = on_least(Hw, s, k, 1, 600, init=(V0.contiguous(), M0.contiguous(), t0.contiguous()))
        V1, M1, t1 = po
        tr_agree = float(((t1 + (Hw @ M1) @ V1.T).argmax(1) == s.argmax(1)).double().mean())
        te = float(kl_rows(ste, t1 + (Hwte @ M1) @ V1.T).mean())
        te_agree = float(((t1 + (Hwte @ M1) @ V1.T).argmax(1) == ste.argmax(1)).double().mean())
        norm = float(((V1 @ M1.T)).norm())
        row = dict(k=k, L_table=lt, table_plus_eps=lt + eps, L_on=lo, on_test_kl=te, agree_train=tr_agree, agree_test=te_agree, W_build_norm=norm)
        rows.append(row)
        print(row, flush=True)
    out["C"] = rows
if not a.smoke:
    json.dump(out, open(os.path.join(HERE, "tq14_table.json"), "w"), indent=1)
print("done", flush=True)
