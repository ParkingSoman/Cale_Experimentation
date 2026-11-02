# serves: TQ13
"""Why the q^theta bound of tq13_check.py came out 0: the best possible bound of TQ13 part 1 at a given R is the least of D over the hull of
{rank <= k, |W|_F <= R}, the k-support ball. Estimator: Frank-Wolfe over that ball (atoms: R x the top-k SVD part of -Gamma, normalised), with t
refit each step; the bound is the Frank-Wolfe dual value, a proved lower bound on the hull's least. Same instances as tq13_check.py (C = 6, d = 5,
k = 1 and 2), R = R* (the norm of the best rank-k head found) and R = |W_N|_F. Prints hull bound / best found.
Prediction, written after seeing the zero bound and before this run: the ratio is well below 1 at R = R*, because a full-rank head of
the same size fits far better than a rank-k head."""
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(11)
C, d, n = 6, 5, 60


def inst():
    z = torch.randn(n, d)
    z = z - z.mean(0)
    Wn = torch.randn(C, d) * 1.2
    y = z @ Wn.T + .3 * torch.randn(n, C)
    return z, torch.softmax(y, 1)


def D(W, t, z, s):
    return (s * (torch.log(s) - torch.log_softmax(z @ W.T + t, 1))).sum(1).mean()


def fit(z, s, k, seed):
    g = torch.Generator().manual_seed(seed)
    V = torch.randn(C, k, generator=g).requires_grad_(True)
    M = torch.randn(k, d, generator=g).requires_grad_(True)
    t = torch.zeros(C).requires_grad_(True)
    opt = torch.optim.LBFGS([V, M, t], lr=1, max_iter=800, history_size=50, line_search_fn="strong_wolfe", tolerance_grad=1e-10,
                            tolerance_change=1e-15)

    def cl():
        opt.zero_grad()
        v = D(V @ M, t, z, s)
        v.backward()
        return v
    opt.step(cl)
    return (V @ M).detach(), t.detach()


def refit_t(W, t0, z, s):
    t = t0.clone().requires_grad_(True)
    o = torch.optim.LBFGS([t], lr=1, max_iter=60, line_search_fn="strong_wolfe", tolerance_grad=1e-11)

    def c():
        o.zero_grad()
        v = D(W, t, z, s)
        v.backward()
        return v
    o.step(c)
    return t.detach()


def atom(G, k, R):
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    A = (U[:, :k] * S[:k]) @ Vh[:k]
    return -R * A / A.norm()


def fw_bound(z, s, k, R, steps=400):
    W = torch.zeros(C, d)
    t = refit_t(W, torch.zeros(C), z, s)
    best = -1e9
    for i in range(steps):
        q = torch.softmax(z @ W.T + t, 1)
        G = (q - s).T @ z / n
        val = float(D(W, t, z, s))
        A = atom(G, k, R)
        gap = float((G * (W - A)).sum())
        best = max(best, val - gap)
        lo, hi = 0.0, 1.0
        for _ in range(30):                    # exact line search on a convex function (golden-free bisection on the slope)
            m = (lo + hi) / 2
            Wm = (1 - m) * W + m * A
            qm = torch.softmax(z @ Wm.T + t, 1)
            slope = float((((qm - s).T @ z / n) * (A - W)).sum())
            lo, hi = (m, hi) if slope < 0 else (lo, m)
        W = (1 - lo) * W + lo * A
        t = refit_t(W, t, z, s)
    return best, float(D(W, t, z, s)), int(torch.linalg.matrix_rank(W, tol=1e-6))


for k in (1, 2):
    rows = []
    for trial in range(6):
        z, s = inst()
        Wb, tb = min((fit(z, s, k, trial * 10 + j) for j in range(6)), key=lambda p: float(D(p[0], p[1], z, s)))
        Dmin = float(D(Wb, tb, z, s))
        Rs = float(Wb.norm())
        lb, ub, rk = fw_bound(z, s, k, Rs)
        rows.append((lb / Dmin, ub / Dmin, rk))
    r = np.array(rows)
    print(f"k={k}: at R = R*: dual bound / best found: mean {r[:, 0].mean():.3f}; hull's least / best found {r[:, 1].mean():.3f}; rank of hull solution {r[:, 2].mean():.1f}")
