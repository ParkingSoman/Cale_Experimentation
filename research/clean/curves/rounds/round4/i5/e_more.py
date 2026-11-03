# serves: TQ31
"""Step 5. Four further checks.

(1) Build G: as build E, but each side link has a cause: an edge unit L_e, on in the rows of both families of edge e, voting for the classes of both. Levels of side links live there.
    PREDICTION: literal eps_V > 0 (two edge units vote opposite ways on a pair of tied side classes). Far below ResNet-18's .91. Other errors 0.
(2) D20: on C5 (rotation), and G + G (swap). Residual of W_out pi = sigma W_out. PREDICTION: 0, so D20 does not separate.
(3) Free-code fit at k = 1 and k = 2, 3 on square tables, with kept offsets. Does the sign count of k = 1 say anything at k = 2? PREDICTION: fit at k = 2 far below k = 1, on sparse graphs.
(4) Competition graph. Edge c - i when i is in the mass at least .01 of row c's off-own table, c likewise in row i, and M' < -.2 in both rows, with M' = m[i]/m[own] - 1, m = first principal direction of the row's movement in its own metric.
    tau = 3 triangles / wedges. Bound: uncut >= triangles / d_max, by pigeonhole, for k = 1. Null: same degrees, edges rewired. On ResNet-18: all links, and side links only (cosW < .1). Signs from half A of the photos, links from half B.
    PREDICTION: hard tables tau near 0 at t = 1 and for side links at any t. ResNet-18 tau well above its null, mostly from family links.
"""
import itertools
import sys

import numpy as np
import torch

HERE = "/Users/shonusengupta/Cale_Experimentation/research/clean/curves"
sys.path.insert(0, f"{HERE}/rounds/round4/i5")
from netlib import *
from sqlib import *

torch.set_default_dtype(torch.float64)
torch.set_num_threads(3)
np.set_printoptions(precision=3, suppress=True, linewidth=160)
rng = np.random.default_rng(23)


def build_G(M, W, fam, edges, a, D):
    n = len(W)
    q = fam.max() + 1
    y0 = np.log(W)
    E = len(edges)
    deg = np.zeros(q, int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    W1 = np.zeros((n + 3 * q + E, 2 * n))
    for i in range(n):
        W1[i, i] = 1
        v = fam[i]
        W1[n + v, i] = 1
        W1[n + q + v, [i, n + i]] = -2, 1
        W1[n + 2 * q + v, [i, n + i]] = 2, -1
    for k, (u, v) in enumerate(edges):
        mem = np.where((fam == u) | (fam == v))[0]
        W1[n + 3 * q + k, mem] = 1
    N2 = n + 3 * q + E
    W2 = np.zeros((N2, N2))
    b2 = np.zeros(N2)
    for i in range(n):
        W2[i, i] = 1
    for v in range(q):
        mem = np.where(fam == v)[0]
        W2[n + v, n + v] = 2
        W2[n + v, mem] = 1
        b2[n + v] = -2
        W2[n + q + v, [n + q + v, n + v]] = 1, 2
        W2[n + q + v, mem] = 1
        W2[n + 2 * q + v, [n + 2 * q + v, n + v]] = 1, 2
        W2[n + 2 * q + v, mem] = 1
        b2[n + q + v] = b2[n + 2 * q + v] = -3
    for k, (u, v) in enumerate(edges):
        mem = np.where((fam == u) | (fam == v))[0]
        W2[n + 3 * q + k, n + 3 * q + k] = 2
        W2[n + 3 * q + k, mem] = 1
        b2[n + 3 * q + k] = -2
    Wout = np.zeros((n, N2))
    l = np.log(D) + a + 1
    for v in range(q):
        mem = np.where(fam == v)[0]
        rep = mem[0]
        pm = a / 2 * M[rep].copy()
        Wout[:, n + q + v] = pm
        Wout[:, n + 2 * q + v] = -pm
        Wout[mem, n + v] = l - deg[v] * np.log(D)
        for i in mem:
            Wout[i, i] = y0[i, i] - l
    for k, (u, v) in enumerate(edges):
        mem = np.where((fam == u) | (fam == v))[0]
        Wout[mem, n + 3 * q + k] = np.log(D)
    return [(W1, np.zeros(W1.shape[0])), (W2, b2)], Wout, np.zeros(n)


print("=== (1) build G")
for a in (1.0, 4.0):
    for name, (qq, e), t in (("C5 t=2", cycle(5), 2), ("C5 t=3", cycle(5), 3), ("C4 t=2", cycle(4), 2)):
        n = qq * t
        D = float(np.exp(np.ceil(np.log(n ** 2 * 50))))
        M, W, fam, adj, Df = sqfam(qq, e, t, a, .31, D)
        Y = table(np.log(W), a / 2 * M)
        lay, Wo, c = build_G(M, W, fam, e, a, D)
        X = points(n)
        p = np.ones(2 * n) / (2 * n)
        err = np.abs(forward(lay, Wo, c, X)[2] - Y).max()
        r = principles(lay, Wo, c, X, p)
        bad = sum(1 for q in r["per"] if q["xi"] is not None and q["xi"] <= 1e-13)
        print(f"a={a} {name}: units {r['units']} table_err={err:.1e} eps_V={r['eps_V']:.4f} V_rel={r['eps_V_rel']:.4f} G={r['eps_G']} B={r['eps_B']} pack={r['eps_pack']:.4f} unread={r['eps_unread']:.3f} agree={r['eps_agree']:.3f} xi<=0 units {bad}/{len(r['per'])} min_lam={min(q['lam'] for q in r['per']):.4f}", flush=True)

print("=== (2) D20")
n, e = cycle(5)
D = float(np.exp(np.ceil(np.log(n ** 2 * 50))))
y0, mv = tq31(n, e, 1.0, .31, D)
layers, Wo, c = net_dq18(y0, mv)
rot = np.roll(np.arange(5), 1)
units = np.r_[rot, 5 + rot]
cls = np.r_[0, 1 + rot, 6 + rot, 11 + rot]
print("C5 rotation, TQ31 table, residual W_out pi - sigma W_out:", np.abs(Wo[:, units] - Wo[np.argsort(cls)][:, :] if False else Wo[np.ix_(cls, units)] - Wo).max())
n2, e2 = cycle(5)
ee = e2 + [(u + 5, v + 5) for u, v in e2]
y0, mv = tq31(10, ee, 1.0, .31, float(np.exp(np.ceil(np.log(100 * 50)))))
layers, Wo, c = net_dq18(y0, mv)
sw = np.r_[np.arange(5, 10), np.arange(5)]
units = np.r_[sw, 10 + sw]
cls = np.r_[0, 1 + sw, 11 + sw, 21 + sw]
print("G + G swap: residual:", np.abs(Wo[np.ix_(cls, units)] - Wo).max(), " U doubles:", least_uncut(10, ee)[0], "=2x", least_uncut(5, e2)[0])

print("=== (3) free fit at k units, kept offsets")


def fit_k(y0, mv, k, starts=5, iters=500):
    best = np.inf
    m, C = y0.shape
    for s in range(starts):
        torch.manual_seed(s)
        t = y0.clone().requires_grad_(True)
        V = (torch.randn(C, k) * .5).requires_grad_(True)
        cp = torch.randn(m, k).requires_grad_(True)
        cm = torch.randn(m, k).requires_grad_(True)
        o = torch.optim.LBFGS([t, V, cp, cm], lr=1, max_iter=iters, history_size=40, line_search_fn="strong_wolfe", tolerance_grad=1e-11, tolerance_change=1e-15)
        yp, ym = y0 + mv, y0 - mv
        sp, sm = torch.softmax(yp, 1), torch.softmax(ym, 1)

        def cl():
            o.zero_grad()
            lp = t + cp @ V.T
            lm = t + cm @ V.T
            v = ((sp * (torch.log_softmax(yp, 1) - torch.log_softmax(lp, 1))).sum() + (sm * (torch.log_softmax(ym, 1) - torch.log_softmax(lm, 1))).sum()) / (2 * m)
            v.backward()
            return v
        o.step(cl)
        best = min(best, float(cl().detach()))
    return best


a = 1.0
for name, (q, e), t in (("C5", cycle(5), 1), ("sub(K4)", subdivide(*complete(4)), 1), ("K5", complete(5), 1), ("K5", complete(5), 2), ("Petersen", (10, [(i, (i + 1) % 5) for i in range(5)] + [(i + 5, (i + 2) % 5 + 5) for i in range(5)] + [(i, i + 5) for i in range(5)]), 1)):
    n = q * t
    U, sig = least_uncut(q, e)
    D = float(np.exp(np.ceil(np.log(n ** 2 * 50))))
    M, W, fam, adj, Df = sqfam(q, e, t, a, .3, D)
    y0 = torch.from_numpy(np.log(W))
    mv = torch.from_numpy(a / 2 * M)
    fs = [fit_k(y0, mv, k) for k in (1, 2, 3)]
    print(f"{name} t={t} n={n} U={U} E={len(e)}: fit k=1 {fs[0]:.5f}, k=2 {fs[1]:.5f}, k=3 {fs[2]:.6f}", flush=True)

print("=== (4) competition graph")


def pc_rows(Y, idx, Wn):
    K, C = Wn.shape
    Mp = np.zeros((K, C))
    for c in range(K):
        d = Y[idx[c]] - Y[idx[c]].mean(0)
        w = Wn[c]
        d = d - (d @ w)[:, None]
        sq = np.sqrt(w)
        U, s, Vt = np.linalg.svd(d * sq[None, :], full_matrices=False)
        m = Vt[0] / np.where(sq > 1e-6, sq, np.inf)
        Mp[c] = m / m[c] - 1 if abs(m[c]) > 1e-9 else np.nan
    return Mp


def comp_stats(T, Mp, mask=None, thr=-.2, minmass=.01):
    K = len(T)
    H = T >= minmass
    R = H & H.T
    compl = (Mp < thr) & (Mp.T < thr)
    A = R & compl
    if mask is not None:
        A = A & mask
    np.fill_diagonal(A, False)
    Ai = A.astype(float)
    deg = Ai.sum(1)
    E = Ai.sum() / 2
    tri = np.trace(Ai @ Ai @ Ai) / 6
    wedges = (deg * (deg - 1) / 2).sum()
    tau = 3 * tri / wedges if wedges else 0.0
    # null: rewire, degree preserving
    ed0 = [tuple(int(z) for z in p) for p in np.array(np.where(np.triu(A))).T]
    taus = []
    for rep in range(5):
        Ed = list(ed0)
        have = set(Ed) | {(b_, a_) for a_, b_ in Ed}
        for sw_ in range(10 * len(Ed)):
            i, j = int(rng.integers(len(Ed))), int(rng.integers(len(Ed)))
            a1, b1 = Ed[i]
            a2, b2 = Ed[j]
            if len({a1, b1, a2, b2}) == 4 and (a1, b2) not in have and (a2, b1) not in have:
                have -= {(a1, b1), (b1, a1), (a2, b2), (b2, a2)}
                have |= {(a1, b2), (b2, a1), (a2, b1), (b1, a2)}
                Ed[i] = (a1, b2)
                Ed[j] = (a2, b1)
        Bm = np.zeros((K, K))
        for a_, b_ in Ed:
            Bm[a_, b_] = Bm[b_, a_] = 1
        assert np.array_equal(Bm.sum(1), deg)
        t3 = np.trace(Bm @ Bm @ Bm) / 6
        taus.append(3 * t3 / wedges if wedges else 0.0)
    # local search cut on the graph
    Aorig = A.astype(float)
    best = 1.0
    for t_ in range(30):
        x = rng.choice((-1., 1.), K)
        for _ in range(200):
            f = Aorig @ x
            fl = np.where(x * f > 0)[0]
            if len(fl) == 0:
                break
            i = fl[np.argmax(np.abs(f[fl]))]
            x[i] = -x[i]
        if E:
            best = min(best, float(((np.outer(x, x) > 0) * Aorig).sum() / 2 / E))
    return dict(links=int(H.sum()), reciprocal=int(R.sum() // 2), comp_edges=int(E), share_comp=float(A.sum() / max(R.sum(), 1)), tau=float(tau), tau_null=float(np.mean(taus)), triangles=int(tri), dmax=int(deg.max()) if len(deg) else 0, uncut_lower=float(tri / deg.max() / E) if E and deg.max() else 0.0, uncut_found=best)


for name, (q, e), t in (("C5", cycle(5), 1), ("C5", cycle(5), 3), ("sub(K5)", subdivide(*complete(5)), 1), ("sub(K5)", subdivide(*complete(5)), 2), ("K5", complete(5), 2)):
    n = q * t
    D = float(np.exp(np.ceil(np.log(n ** 2 * 50))))
    M, W, fam, adj, Df = sqfam(q, e, t, 1.0, .3, D)
    Y = table(np.log(W), .5 * M)
    idx = [np.array([2 * c, 2 * c + 1]) for c in range(n)]
    S = (softmax(Y[0::2]) + softmax(Y[1::2])) / 2
    T = S.copy()
    np.fill_diagonal(T, 0)
    T = T / T.sum(1, keepdims=True)
    Mp = pc_rows(Y, idx, S)
    print(f"{name} t={t}:", comp_stats(T, Mp), flush=True)

Yall = np.load(f"{HERE}/cache/map/resnet18_y.npy").astype(np.float64)
meta = np.load(f"{HERE}/cache/map/resnet18_meta.npz")
lab = meta["labels"].astype(int)
Wout = meta["W"].astype(np.float64)
K = 1000
Sx = softmax(Yall)
gen = np.random.default_rng(7)
ia, ib = [], []
for c in range(K):
    ix = gen.permutation(np.where(lab == c)[0])
    ia.append(ix[:10])
    ib.append(ix[10:])
SA = np.stack([Sx[i].mean(0) for i in ia])
SB = np.stack([Sx[i].mean(0) for i in ib])
TB = SB.copy()
np.fill_diagonal(TB, 0)
TB = TB / TB.sum(1, keepdims=True)
Mp = pc_rows(Yall, ia, SA)
un = Wout / np.linalg.norm(Wout, axis=1, keepdims=True)
cosW = un @ un.T
print("ResNet-18 all links:", comp_stats(TB, Mp), flush=True)
print("ResNet-18 family links (cosW >= .3):", comp_stats(TB, Mp, cosW >= .3), flush=True)
print("ResNet-18 side links (cosW < .1):", comp_stats(TB, Mp, cosW < .1), flush=True)
print("ResNet-18 at mass >= .003, all links:", comp_stats(TB, Mp, None, -.2, .003), flush=True)
print("ResNet-18 at mass >= .03, all links:", comp_stats(TB, Mp, None, -.2, .03), flush=True)
# sign reproducibility of M' between halves, on the head
Mp2 = pc_rows(Yall, ib, SB)
top = np.argsort(-TB, 1)[:, :8]
a1 = Mp[np.arange(K)[:, None], top].ravel()
b1 = Mp2[np.arange(K)[:, None], top].ravel()
okk = np.isfinite(a1) & np.isfinite(b1)
print("M' sign agreement between halves on head links:", float((np.sign(a1[okk]) == np.sign(b1[okk])).mean()), " share M' < -.2:", float((a1[okk] < -.2).mean()))
