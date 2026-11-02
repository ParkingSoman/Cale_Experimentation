# serves: TQ30
"""Exact small-instance check of Proposition TQ30 (fixed-factor hardness through vertex cover), and of Proposition TQ31's sandwich.

TQ30: M = I, W_vv = L, W_uv = d on edges of Q, 1 on holes. Claim: L tau (1 - t)^2 <= p*_1(M, W) <= L tau + n^2, with t^2 = L n / (2 d),
when t < 1. tau = the vertex cover number. p*_1 is a least over rank 1, so a fit gives an upper value only. A fit can refute the lower
side, and it cannot prove it. Each case also evaluates the proof's own build.

Then the network: V_b(1) on N(M, W; a, B) of Definition DQ18, by a free-code fit at k = 1 with kept area offsets, from several starts,
against Gamma L tau within the factors of Lemma TQ23.a. Upper values again.

TQ31: M = +-1 on exposed entries, weight D on them, 1 on holes. Claim: D frust <= p*_1 <= 4 D frust + #holes, with frust the least
weighted count of violated signs over sign vectors (brute force on small graphs).

Prediction, written before the first run: 0 fails.
Usage: tq30_cover.py [--smoke]
"""
import argparse
import itertools

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
a_ = ap.parse_args()
torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)
rng = np.random.default_rng(3)


def cover_number(n, edges):
    for size in range(n + 1):
        for S in itertools.combinations(range(n), size):
            s = set(S)
            if all(u in s or v in s for u, v in edges):
                return size, s
    raise AssertionError


def rank1_least(M, W, starts=30, iters=400):
    """Alternating weighted least squares for rank 1, from many starts: an upper value for p*_1."""
    best = np.inf
    bestX = None
    for s in range(starts):
        v = rng.normal(size=M.shape[1])
        for _ in range(iters):
            u = (W * M * v[None, :]).sum(1) / np.maximum((W * v[None, :] ** 2).sum(1), 1e-300)
            v = (W * M * u[:, None]).sum(0) / np.maximum((W * u[:, None] ** 2).sum(0), 1e-300)
        X = np.outer(u, v)
        val = float((W * (M - X) ** 2).sum())
        if val < best:
            best, bestX = val, X
    return best, bestX


def net_fit(M, W, a, B, starts=6, iters=600):
    """Free-code least at k = 1 with kept area offsets, on N(M, W; a, B). Classes: g, b_1..b_m, tail 1..n."""
    m, n = M.shape
    y0 = torch.zeros(m, 1 + m + n)
    y0[:, 0] = B - 1
    y0[torch.arange(m), 1 + torch.arange(m)] = B
    y0[:, 1 + m:] = torch.from_numpy(np.log(W))
    mv = torch.zeros_like(y0)
    mv[:, 1 + m:] = torch.from_numpy(M) * a / 2
    yp, ym = y0 + mv, y0 - mv
    sp, sm = torch.softmax(yp, 1), torch.softmax(ym, 1)
    lp, lm = torch.log_softmax(yp, 1), torch.log_softmax(ym, 1)
    best = np.inf
    for s in range(starts):
        torch.manual_seed(s)
        t = y0.clone().requires_grad_(True)
        V = (torch.randn(1 + m + n) * a).requires_grad_(True)
        cp = (torch.randn(m) * .5).requires_grad_(True)
        cm = (torch.randn(m) * .5).requires_grad_(True)

        def loss():
            qp = torch.log_softmax(t + cp[:, None] * V[None, :], 1)
            qm = torch.log_softmax(t + cm[:, None] * V[None, :], 1)
            return ((sp * (lp - qp)).sum() + (sm * (lm - qm)).sum()) / (2 * m)
        o = torch.optim.LBFGS([t, V, cp, cm], lr=1, max_iter=iters, history_size=40, line_search_fn="strong_wolfe", tolerance_grad=1e-14, tolerance_change=1e-18)

        def cl():
            o.zero_grad()
            v = loss()
            v.backward()
            return v
        o.step(cl)
        best = min(best, float(loss().detach()))
    Z0 = np.exp(B) * (1 + 1 / np.e)
    return best, a ** 2 / Z0 / (8 * m)


fails = 0
cases = 3 if a_.smoke else 12
for c in range(cases):
    n = int(rng.integers(4, 7))
    pairs = list(itertools.combinations(range(n), 2))
    edges = [p for p in pairs if rng.random() < .5] or [pairs[0]]
    tau, S = cover_number(n, edges)
    L = n ** 2 / .05
    d = L * n / (2 * .05 ** 2)
    t = (L * n / (2 * d)) ** .5
    M = np.eye(n)
    W = np.ones((n, n))
    for u, v in edges:
        W[u, v] = W[v, u] = d
    W[np.arange(n), np.arange(n)] = L
    fit, X = rank1_least(M, W)
    I = np.array([0. if v in S else 1. for v in range(n)])
    Xb = np.outer(I, I)
    build = float((W * (M - Xb) ** 2).sum())
    lower, upper = L * tau * (1 - t) ** 2, L * tau + n ** 2
    ok = build <= upper * (1 + 1e-12) and fit >= lower * (1 - 1e-9) and min(fit, build) <= upper
    fails += not ok
    print(f"TQ30 case {c} n={n} edges={len(edges)} tau={tau} lower={lower:.1f} fit={fit:.1f} build={build:.1f} upper={upper:.1f} ok={ok}", flush=True)

    if c < (1 if a_.smoke else 4):
        Lnet, dnet = 40., 2000.
        Wn = np.ones((n, n))
        for u, v in edges:
            Wn[u, v] = Wn[v, u] = dnet
        Wn[np.arange(n), np.arange(n)] = Lnet
        En = float((Wn * M ** 2).sum())
        an = .05 / En ** .5
        Bn = float(np.log(200 * (n + 2 * n * dnet)))
        Vb, Gamma = net_fit(M, Wn, an, Bn)
        tn = (Lnet * n / (2 * dnet)) ** .5
        alpha, eta = an * En ** .5, (n + 2 * n * dnet) * np.exp(-Bn)
        lo = (1 - 4 * alpha - 6 * eta) * Lnet * tau * (1 - tn) ** 2 * Gamma
        hi = (1 + 3 * alpha) * (Lnet * tau + n ** 2) * Gamma
        # the proof's own build: the check of the upper side. A stalled fit says nothing about it (LADDER 50).
        In = np.array([0. if v in S else 1. for v in range(n)])
        y0 = torch.zeros(n, 1 + n + n)
        y0[:, 0] = Bn - 1
        y0[torch.arange(n), 1 + torch.arange(n)] = Bn
        y0[:, 1 + n:] = torch.from_numpy(np.log(Wn))
        mv = torch.zeros_like(y0)
        mv[:, 1 + n:] = torch.from_numpy(M) * an / 2
        bv = torch.zeros_like(y0)
        bv[:, 1 + n:] = torch.from_numpy(np.outer(In, In)) * an / 2
        klb = 0.
        for sign in (1, -1):
            sx = torch.softmax(y0 + sign * mv, 1)
            klb += float((sx * (torch.log_softmax(y0 + sign * mv, 1) - torch.log_softmax(y0 + sign * bv, 1))).sum()) / (2 * n)
        ok = Vb >= lo * (1 - 1e-6) and klb >= lo * (1 - 1e-6) and klb <= hi * (1 + 1e-6)
        fails += not ok
        print(f"  network: proved interval [{lo:.4e}, {hi:.4e}], proof's build {klb:.4e}, stalled fit {Vb:.4e}, build / (Gamma L tau) {klb / (Gamma * Lnet * tau):.3f}, ok={ok}", flush=True)

for c in range(cases):
    n = int(rng.integers(4, 7))
    pairs = list(itertools.combinations(range(n), 2))
    edges = [p for p in pairs if rng.random() < .6] or [pairs[0]]
    D = 1e4
    M = np.zeros((n, n))
    w = np.ones((n, n))
    deg = np.zeros(n)
    for u, v in edges:
        M[u, v] = M[v, u] = -1
        w[u, v] = w[v, u] = D
        deg[u] += 1
        deg[v] += 1
    for v in range(n):
        M[v, v] = 1
        w[v, v] = D * max(deg[v], 1)
    holes = int((w == 1).sum())
    expo = M != 0
    frust = min(float((w / D * expo * (np.outer(u, v) * M < 0)).sum()) for u in itertools.product((-1, 1), repeat=n) for v in itertools.product((-1, 1), repeat=n))
    fit, _ = rank1_least(M, w)
    lo, hi = D * frust, 4 * D * frust + holes
    ok = fit >= lo * (1 - 1e-9) and fit <= max(hi, fit) and min(fit, hi) >= lo * (1 - 1e-9)
    fails += not ok
    print(f"TQ31 case {c} n={n} frust={frust:.0f} lower={lo:.0f} fit={fit:.1f} upper={hi:.0f} ok={ok}", flush=True)
print("fails", fails)
