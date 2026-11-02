# serves: TQ31
"""Check of Proposition TQ31 part 2 at real size: any movement a, any tail mass rho, at ordinary loss levels, so a free fit tests something.

Small graphs, U(G) by brute force. The network of TQ31: classes g, b_1..b_n, the n x n block's columns, filler columns. Rows: 2 points per area.
Free-code build at k = 1 with kept area offsets, many starts, float64. A fit gives an upper value of L_free(1), so it can refute the lower side, not prove it.
The proof's own build gives an upper value of L_on(1), to be held against the upper side.
Also: do the two sides separate instances by U(G)? Spearman of U(G) against the best fit.

Prediction, written before the first run: 0 fails. The fit lies far above the lower side, since the constant c tau' psi(-a/4) is small (a large constant gap, as for a Markov step).

Usage: tq31_anytail.py [--smoke]
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
rng = np.random.default_rng(5)


def least_uncut(n, edges):
    return min(sum(1 for u, v in edges if s[u] == s[v]) for s in itertools.product((0, 1), repeat=n))


def network(n, edges, a, rho, D):
    deg = np.zeros(n)
    M = np.zeros((n, n))
    W = np.ones((n, n))
    for u, v in edges:
        M[u, v] = M[v, u] = -1
        W[u, v] = W[v, u] = D
        deg[u] += 1
        deg[v] += 1
    for v in range(n):
        M[v, v] = 1
        W[v, v] = D * max(deg[v], 1)
    rowsum = W.sum(1)
    S = rowsum.max() + n
    fill = rowsum.max() - rowsum  # one filler column per row level: a diagonal block keeps the classes few
    Fill = np.ones((n, n)) + np.diag(fill)
    Wall = np.concatenate([W, Fill], 1)
    Mall = np.concatenate([M, np.zeros((n, n))], 1)
    # heavy part so that the tail holds about rho of each row
    heavy = S * (1 - rho) / rho
    B = float(np.log(heavy / (1 + 1 / np.e)))
    return Mall, Wall, B


def tables(M, W, a, B):
    m, nt = M.shape
    y0 = torch.zeros(m, 1 + m + nt)
    y0[:, 0] = B - 1
    y0[torch.arange(m), 1 + torch.arange(m)] = B
    y0[:, 1 + m:] = torch.from_numpy(np.log(W))
    mv = torch.zeros_like(y0)
    mv[:, 1 + m:] = torch.from_numpy(M) * a / 2
    return y0, mv


def kl_pts(y0, mv, lp, lm):
    sp, sm = torch.softmax(y0 + mv, 1), torch.softmax(y0 - mv, 1)
    return ((sp * (torch.log_softmax(y0 + mv, 1) - torch.log_softmax(lp, 1))).sum() + (sm * (torch.log_softmax(y0 - mv, 1) - torch.log_softmax(lm, 1))).sum()) / (2 * y0.shape[0])


def free_fit(y0, mv, starts, iters):
    best = np.inf
    C = y0.shape[1]
    m = y0.shape[0]
    for s in range(starts):
        torch.manual_seed(s)
        t = y0.clone().requires_grad_(True)
        V = (torch.randn(C) * .5).requires_grad_(True)
        cp = torch.randn(m).requires_grad_(True)
        cm = torch.randn(m).requires_grad_(True)
        o = torch.optim.LBFGS([t, V, cp, cm], lr=1, max_iter=iters, history_size=40, line_search_fn="strong_wolfe", tolerance_grad=1e-11, tolerance_change=1e-15)

        def cl():
            o.zero_grad()
            v = kl_pts(y0, mv, t + cp[:, None] * V[None, :], t + cm[:, None] * V[None, :])
            v.backward()
            return v
        o.step(cl)
        best = min(best, float(cl().detach()))
    return best


def g_of(c):
    """g in Lemma TQ8 form: KL <= g(c) (1/2) Var, with g(c) = 2 (e^c - 1 - c) / c^2 at spread c."""
    return 2 * (np.exp(c) - 1 - c) / c ** 2


fails = 0
rows = []
for a, rho in ((1.0, .31), (4.0, .31), (4.0, .05)):
    uu, ff = [], []
    for c in range(2 if args.smoke else 10):
        n = int(rng.integers(4, 7))
        pairs = list(itertools.combinations(range(n), 2))
        edges = [p for p in pairs if rng.random() < .6] or [pairs[0]]
        # no isolated vertices are needed for the fit; the proof drops them. Keep them only with W_vv set as in network().
        U = least_uncut(n, edges)
        D = float(np.exp(np.ceil(np.log(max(n ** 2, 2) * 50))))
        M, W, B = network(n, edges, a, rho, D)
        y0, mv = tables(M, W, a, B)
        s_p, s_m = torch.softmax(y0 + mv, 1), torch.softmax(y0 - mv, 1)
        smin = torch.minimum(s_p, s_m)
        smax = torch.maximum(s_p, s_m)
        Wt = torch.from_numpy(W)
        off = 1 + n
        block = Wt[:, :n]
        taup = float((smin[:, off:off + n] / block).min())
        taupp = float((smax[:, off:off + n] / block).max())
        s0 = float(smin[:, 0].min())
        Sp = float(block.sum(1).max())
        cc = min(1., s0 / (taup * Sp))
        psi = np.exp(-a / 4) - 1 + a / 4
        lower = cc * taup * psi * D * U / n
        # the proof's own build: X = sigma sigma^T on a best two-siding
        best_s = min(itertools.product((-1, 1), repeat=n), key=lambda s: sum(1 for u, v in edges if s[u] == s[v]))
        sig = np.array(best_s, dtype=float)
        X = np.zeros_like(M)
        X[:, :n] = np.outer(sig, sig)
        bv = torch.zeros_like(y0)
        bv[:, off:] = torch.from_numpy(X) * a / 2
        build = float(kl_pts(y0, mv, y0 + bv, y0 - bv))
        upper = g_of(2 * a) * taupp * (a ** 2 / (8 * n)) * (8 * D * U + n ** 2)
        fit = free_fit(y0, mv, 2 if args.smoke else 8, 200 if args.smoke else 800)
        ok = fit >= lower * (1 - 1e-6) and build <= upper * (1 + 1e-6)
        fails += not ok
        tail = float(s_p[:, off:].sum(1).mean())
        uu.append(U)
        ff.append(fit)
        print(f"a={a} rho={rho} n={n} U={U} tail={tail:.3f} lower={lower:.2e} fit={fit:.4f} build={build:.4f} upper={upper:.3f} ok={ok}", flush=True)
    if len(set(uu)) > 1:
        ranks = lambda x: np.argsort(np.argsort(x))
        print(f"a={a} rho={rho}: Spearman U against fit {np.corrcoef(ranks(uu), ranks(ff))[0, 1]:.2f}", flush=True)
print("fails", fails)
