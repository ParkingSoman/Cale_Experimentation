# serves: TQ31
"""Check of Proposition TQ31 part 2, per-area form (Definition DQ24), on small graphs. A free fit tests the lower side only.

The network: W_vv in [D deg(v), e D deg(v)], W_uv = D on edges, 1 on holes, M = +1, -1, 0. B_w and p_w per area, set from S_w, with kappa chosen for a target tail.
Checks, per graph: every row's tail mass and moving share; the proved interval against the best free fit (8 starts) and the proof's own build; and that the value tracks U(G) / E.

Prediction, written before the first run: 0 fails. Tail mass nearly equal across rows at each a. Moving share above .99. Fit / (U / E) nearly constant across graphs at each a (spread within a factor 2).

Usage: tq31_perarea.py [--smoke]
"""
import argparse
import itertools

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
torch.set_default_dtype(torch.float64)
torch.set_num_threads(3)
rng = np.random.default_rng(8)


def build(n, edges, kappa):
    deg = np.zeros(n, dtype=int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    keep = deg > 0
    idx = -np.ones(n, dtype=int)
    idx[keep] = np.arange(keep.sum())
    n2 = int(keep.sum())
    edges2 = [(idx[u], idx[v]) for u, v in edges]
    deg = deg[keep]
    q = int(np.ceil(np.log(max(n2 ** 2 / max(len(edges2), 1), n2) * 200)))
    D = float(np.exp(q))
    M = np.zeros((n2, n2))
    W = np.ones((n2, n2))
    for u, v in edges2:
        M[u, v] = M[v, u] = -1
        W[u, v] = W[v, u] = D
    for v in range(n2):
        M[v, v] = 1
        W[v, v] = np.exp(np.ceil(np.log(D * deg[v])))
    S = W.sum(1)
    B = np.round(np.log(kappa * S / (1 + 1 / np.e)))
    p = S / S.sum()
    return n2, edges2, D, M, W, S, B, p


def kl(y0, mv, lp, lm, p):
    sp, sm = torch.softmax(y0 + mv, 1), torch.softmax(y0 - mv, 1)
    a = (sp * (torch.log_softmax(y0 + mv, 1) - torch.log_softmax(lp, 1))).sum(1)
    b = (sm * (torch.log_softmax(y0 - mv, 1) - torch.log_softmax(lm, 1))).sum(1)
    return (p * (a + b) / 2).sum()


def free_fit(y0, mv, p, starts, iters):
    best = np.inf
    m, C = y0.shape
    for s in range(starts):
        torch.manual_seed(s)
        t = y0.clone().requires_grad_(True)
        V = (torch.randn(C) * .5).requires_grad_(True)
        cp = torch.randn(m).requires_grad_(True)
        cm = torch.randn(m).requires_grad_(True)
        o = torch.optim.LBFGS([t, V, cp, cm], lr=1, max_iter=iters, history_size=40, line_search_fn="strong_wolfe", tolerance_grad=1e-11, tolerance_change=1e-15)

        def cl():
            o.zero_grad()
            v = kl(y0, mv, t + cp[:, None] * V[None, :], t + cm[:, None] * V[None, :], p)
            v.backward()
            return v
        o.step(cl)
        best = min(best, float(cl().detach()))
    return best


def g_of(c):
    return 2 * (np.exp(c) - 1 - c) / c ** 2


fails = 0
for a, kappa in ((1.0, 2.2), (4.0, 8.0), (4.0, 30.0)):
    rat = []
    for c in range(3 if args.smoke else 12):
        n = int(rng.integers(5, 8))
        pairs = list(itertools.combinations(range(n), 2))
        edges = [e for e in pairs if rng.random() < .6] or [pairs[0]]
        n2, edges2, D, M, W, S, B, p = build(n, edges, kappa)
        E = len(edges2)
        U = min(sum(1 for u, v in edges2 if s[u] == s[v]) for s in itertools.product((-1, 1), repeat=n2))
        y0 = torch.zeros(n2, 1 + n2 + n2)
        y0[:, 0] = torch.from_numpy(B) - 1
        y0[torch.arange(n2), 1 + torch.arange(n2)] = torch.from_numpy(B)
        y0[:, 1 + n2:] = torch.from_numpy(np.log(W))
        mv = torch.zeros_like(y0)
        mv[:, 1 + n2:] = torch.from_numpy(M) * a / 2
        P = torch.from_numpy(p)
        sp, sm = torch.softmax(y0 + mv, 1), torch.softmax(y0 - mv, 1)
        tail = sp[:, 1 + n2:].sum(1)
        moving = (sp[:, 1 + n2:] * (mv[:, 1 + n2:] != 0)).sum(1) / tail
        smin, smax = torch.minimum(sp, sm), torch.maximum(sp, sm)
        Wt, St = torch.from_numpy(W), torch.from_numpy(S)
        tp = float((smin[:, 1 + n2:] * St[:, None] / Wt).min())
        tpp = float((smax[:, 1 + n2:] * St[:, None] / Wt).max())
        s0 = float(smin[:, 0].min())
        cc = min(1., s0 / tp)
        psi = np.exp(-a / 4) - 1 + a / 4
        lower = cc * tp * psi * D * U / S.sum()
        upper = g_of(2 * a) * tpp * (a ** 2 / 8) * (8 * D * U + n2 ** 2) / S.sum()
        best_s = min(itertools.product((-1, 1), repeat=n2), key=lambda s: sum(1 for u, v in edges2 if s[u] == s[v]))
        sig = np.array(best_s, dtype=float)
        bv = torch.zeros_like(y0)
        bv[:, 1 + n2:] = torch.from_numpy(np.outer(sig, sig)) * a / 2
        bld = float(kl(y0, mv, y0 + bv, y0 - bv, P))
        fit = free_fit(y0, mv, P, 2 if args.smoke else 8, 200 if args.smoke else 800)
        ok = fit >= lower * (1 - 1e-6) and bld <= upper * (1 + 1e-6)
        fails += not ok
        if U:
            rat.append(fit / (U / E))
        print(f"a={a} kappa={kappa} n={n2} E={E} U={U} tail {float(tail.min()):.3f}-{float(tail.max()):.3f} moving>={float(moving.min()):.4f} lower={lower:.2e} fit={fit:.4f} build={bld:.4f} upper={upper:.3f} ok={ok}", flush=True)
    if rat:
        print(f"a={a} kappa={kappa}: fit / (U/E) from {min(rat):.3f} to {max(rat):.3f}", flush=True)
print("fails", fails)
