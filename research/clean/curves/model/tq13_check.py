# serves: TQ13
"""Check of Proposition TQ13 on small random instances. C = 6 classes, d = 5 features, k = 2.
Part 1: for random q with the mean moment condition, and random and optimised heads W of rank <= k with |W|_F <= R, the bound never exceeds
D(W, t) (for t optimal, the lowest D for that W). Fails if bound > D + 1e-9.
Part 2: at a rank-k stationary point found by alternating fits: the mean entropy gap equals D to 1e-6, and Gamma W^T, W^T Gamma are near 0.
Part 3: the bound with q^theta, best theta, against the lowest D found over many starts, as R grows: the bound must fall towards 0."""
import json
import os

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(11)
C, d, k, n = 6, 5, 2, 60


def inst():
    z = torch.randn(n, d)
    z = z - z.mean(0)
    Wn = torch.randn(C, d) * 1.2
    y = z @ Wn.T + .3 * torch.randn(n, C)
    return z, torch.softmax(y, 1)


def D(W, t, z, s):
    return (s * (torch.log(s) - torch.log_softmax(z @ W.T + t, 1))).sum(1).mean()


def Hn(q):
    return -(q * torch.log(q.clamp_min(1e-300))).sum(1)


def fit(z, s, k, seed, iters=800):
    g = torch.Generator().manual_seed(seed)
    V = torch.randn(C, k, generator=g).requires_grad_(True)
    M = torch.randn(k, d, generator=g).requires_grad_(True)
    t = torch.zeros(C).requires_grad_(True)
    opt = torch.optim.LBFGS([V, M, t], lr=1, max_iter=iters, history_size=50, line_search_fn="strong_wolfe", tolerance_grad=1e-10,
                            tolerance_change=1e-15)

    def cl():
        opt.zero_grad()
        v = D(V @ M, t, z, s)
        v.backward()
        return v
    opt.step(cl)
    return (V @ M).detach(), t.detach()


def norm_k(G, k):
    sv = torch.linalg.svdvals(G)
    return float((sv[:k] ** 2).sum().sqrt())


fails1, worst2 = 0, 0.0
rows = []
for trial in range(20):
    z, s = inst()
    W, t = min((fit(z, s, k, trial * 10 + j) for j in range(6)), key=lambda p: float(D(p[0], p[1], z, s)))
    Dmin = float(D(W, t, z, s))
    q = torch.softmax(z @ W.T + t, 1)
    gap2 = float((Hn(q) - Hn(s)).mean())
    G = (q - s).T @ z / n
    worst2 = max(worst2, abs(gap2 - Dmin), float((G @ W.T).abs().max()), float((W.T @ G).abs().max()))
    Rstar = float(W.norm())
    for R in (Rstar, 2 * Rstar, 8 * Rstar):
        best = 0.0
        for th in np.linspace(0, 1, 21):
            qt = s + float(th) * (q - s)
            Gt = (qt - s).T @ z / n
            best = max(best, float((Hn(qt) - Hn(s)).mean()) - R * norm_k(Gt, k))
        rows.append(dict(trial=trial, R_over_Rstar=R / Rstar, bound=best, Dmin=Dmin))
        # no head of rank <= k with |W|_F <= R falls below the bound
        for j in range(30):
            A = (torch.randn(C, k) @ torch.randn(k, d))
            A = A * (R * float(torch.rand(1)) / A.norm())
            # t optimal for this head
            tt = torch.zeros(C).requires_grad_(True)
            o = torch.optim.LBFGS([tt], lr=1, max_iter=100, line_search_fn="strong_wolfe")

            def c2():
                o.zero_grad()
                v = D(A, tt, z, s)
                v.backward()
                return v
            o.step(c2)
            if best > float(D(A, tt.detach(), z, s)) + 1e-9:
                fails1 += 1
        if best > Dmin + 1e-9:
            fails1 += 1
print("part 1 fails", fails1, "; part 2 worst stationarity and value error", worst2)
for f in (1, 2, 8):
    r = [x for x in rows if abs(x["R_over_Rstar"] - f) < 1e-6]
    print(f"R = {f} x R*: mean bound / Dmin = {np.mean([x['bound'] / x['Dmin'] for x in r]):.3f}")
json.dump(dict(fails1=fails1, worst2=worst2, rows=rows), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tq13_check.json"), "w"))
