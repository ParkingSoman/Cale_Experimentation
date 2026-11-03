# serves: TG1
"""Exact small check of Lemma TG1, Proposition TG2 and Lemma TG3 (card 3-40). It can only refute.

PREDICTION, written before the first run (2026-11-03): 0 fails. Every instance solves to gradient under 1e-7.
"""
import sys

import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
fails = 0


def check(name, ok, detail=""):
    global fails
    if not ok:
        fails += 1
        print("FAIL", name, detail)


def H(q):
    return -(q * torch.log(q)).sum(-1)


def loss(y, s, p):
    ls = torch.log(s)
    return (p * (s * (ls - torch.log_softmax(y, 1))).sum(1)).sum()


def fit_write(c, s, p, V0=None, t0=None):
    """The convex problem of TG1 at a fixed code."""
    C, k = s.shape[1], c.shape[1]
    V = (torch.randn(C, k) if V0 is None else V0.clone()).requires_grad_(True)
    t = (torch.randn(C) if t0 is None else t0.clone()).requires_grad_(True)
    o = torch.optim.LBFGS([V, t], lr=1, max_iter=4000, history_size=50, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-16)

    def cl():
        o.zero_grad()
        v = loss(c @ V.T + t, s, p)
        v.backward()
        return v

    o.step(cl)
    return V.detach(), t.detach()


def grad(c, V, t, s, p):
    q = torch.softmax(c @ V.T + t, 1)
    w = p[:, None] * (q - s)
    return w.T @ c, w.sum(0), q


def instance(seed):
    torch.manual_seed(seed)
    n, d, C, k = 14, 6, 5, 2
    z = torch.randn(n, d)
    y = torch.relu(z @ torch.randn(d, 9)) @ torch.randn(9, C) * 2
    s = torch.softmax(y, 1)
    p = torch.rand(n) + .05
    p = p / p.sum()
    return z, y, s, p, k


for seed in range(25):
    z, y, s, p, k = instance(seed)
    M = torch.randn(k, z.shape[1])
    c = z @ M.T
    vals = []
    for _ in range(3):
        V, t = fit_write(c, s, p)
        vals.append(float(loss(c @ V.T + t, s, p)))
    check("TG1.1-2 one value from random starts", max(vals) - min(vals) < 1e-8, (seed, vals))
    GV, Gt, q = grad(c, V, t, s, p)
    gn = float((GV.norm() ** 2 + Gt.norm() ** 2).sqrt())
    check("TG1.3 gradient 0 at the fit", gn < 1e-7, (seed, gn))
    dual = float((p * (H(q) - H(s))).sum())
    check("TG1.3 value = entropy gap", abs(dual - vals[-1]) < 1e-7, (seed, dual, vals[-1]))
    for a in (.25, .5, .75):
        r = (1 - a) * s + a * q
        lb = float((p * (H(r) - H(s))).sum())
        check("TG1.4 dual bound below the least", lb <= vals[-1] + 1e-7, (seed, a, lb, vals[-1]))
    V1, t1 = torch.randn_like(V), torch.randn_like(t)
    GV1, Gt1, _ = grad(c, V1, t1, s, p)
    gap = float(loss(c @ V1.T + t1, s, p)) - vals[-1]
    rhs = float((GV1.norm() ** 2 + Gt1.norm() ** 2).sqrt() * ((V1 - V).norm() ** 2 + (t1 - t).norm() ** 2).sqrt())
    check("TG1.5 gap bound", gap <= rhs + 1e-7, (seed, gap, rhs))

    # fixed write: convex in (M, t), so random starts agree
    Vf = torch.randn(s.shape[1], k)
    vals = []
    for _ in range(3):
        Mv = torch.randn(k, z.shape[1]).requires_grad_(True)
        tv = torch.randn(s.shape[1]).requires_grad_(True)
        o = torch.optim.LBFGS([Mv, tv], lr=1, max_iter=4000, history_size=50, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-16)

        def cl():
            o.zero_grad()
            v = loss(z @ Mv.T @ Vf.T + tv, s, p)
            v.backward()
            return v

        o.step(cl)
        vals.append(float(loss(z @ Mv.T @ Vf.T + tv, s, p)))
    check("TG1.1 fixed write: one value", max(vals) - min(vals) < 1e-8, (seed, vals))

    # TG2 on F = all writes at this code (convex, so its minimisers are global)
    A, B = torch.arange(0, 7), torch.arange(7, 14)
    pA, pB = p[A] / p[A].sum(), p[B] / p[B].sum()
    vA = lambda V, t: float(loss(c[A] @ V.T + t, s[A], pA))
    vB = lambda V, t: float(loss(c[B] @ V.T + t, s[B], pB))
    VA, tA = fit_write(c[A], s[A], pA)
    last = None
    for lam in (0., .1, .3, .5, .7, .9):
        pl = torch.cat([(1 - lam) * pA, lam * pB]) if lam > 0 else torch.cat([pA, torch.zeros_like(pB) + 1e-300])
        V, t = fit_write(torch.cat([c[A], c[B]]), torch.cat([s[A], s[B]]), pl)
        a, b = vA(V, t), vB(V, t)
        check("TG2.1 bound", a <= vA(VA, tA) + lam / (1 - lam) * vB(VA, tA) + 1e-6, (seed, lam))
        if last is not None:
            check("TG2.2 exchange", last[0] <= a + 1e-6 and b <= last[1] + 1e-6, (seed, lam, last, a, b))
        last = (a, b)

    # TG3: n <= k + 1 inputs, written directly
    zs, ys = z[:k + 1], y[:k + 1]
    D = (zs[1:] - zs[0]).T  # d x (n-1)
    Ms = torch.linalg.pinv(D)  # k x d, Ms D = I
    Vs = (ys[1:] - ys[0]).T
    ts = ys[0] - Vs @ Ms @ zs[0]
    yb = zs @ Ms.T @ Vs.T + ts
    k0 = float((torch.softmax(ys, 1) * (torch.log_softmax(ys, 1) - torch.log_softmax(yb, 1))).sum(1).abs().max())
    check("TG3 loss 0 on the short list", k0 < 1e-9, (seed, k0))

print("fails", fails)
sys.exit(1 if fails else 0)
