# serves: TQ31
"""Step 3. Does the square family table stay hard? No class g, no filler, own class is the reference, families in the head.

Fixed before the run. Small graphs on q vertices, family size t = 1, 2. Free-code build at k = 1 with kept area offsets, as in model/tq31_anytail.py.
Check: lower <= fit, and the build X = sig sig^T <= upper. Also the fit against U(G): if it tracks U, the table's cost lives in the uncut links.
A fit is an upper value of L_free(1). So it can refute the lower side, not prove it.
Also, holes M = sig sig^T for a two-siding sig (table of build F): same check.
PREDICTION: 0 fails. Fit rises with U at fixed graph size, and at U = 0 the fit is on the order of the hole term only.
"""
import itertools
import sys

import numpy as np
import torch

sys.path.insert(0, "/Users/shonusengupta/Cale_Experimentation/research/clean/curves/rounds/round4/i5")
from netlib import cycle, complete, least_uncut, subdivide
from sqlib import *

torch.set_default_dtype(torch.float64)
torch.set_num_threads(3)
smoke = "--smoke" in sys.argv
rng = np.random.default_rng(9)


def kl_pts(y0, mv, lp, lm):
    sp, sm = torch.softmax(y0 + mv, 1), torch.softmax(y0 - mv, 1)
    return ((sp * (torch.log_softmax(y0 + mv, 1) - torch.log_softmax(lp, 1))).sum() + (sm * (torch.log_softmax(y0 - mv, 1) - torch.log_softmax(lm, 1))).sum()) / (2 * y0.shape[0])


def free_fit(y0, mv, starts, iters):
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
            v = kl_pts(y0, mv, t + cp[:, None] * V[None, :], t + cm[:, None] * V[None, :])
            v.backward()
            return v
        o.step(cl)
        best = min(best, float(cl().detach()))
    return best


def run(tag, q, e, t, a, rho, hole=False):
    U, sig = least_uncut(q, e)
    n = q * t
    D = float(np.exp(np.ceil(np.log(n ** 2 * 50))))
    M, W, fam, adj, Df = sqfam(q, e, t, a, rho, D, hole_sign=sig if hole else None)
    y0 = torch.from_numpy(np.log(W))
    mv = torch.from_numpy(a / 2 * M)
    lo, up = bounds(M, W, fam, a, D, t, U)
    s = sig[fam]
    X = torch.from_numpy(np.outer(s, s))
    build = float(kl_pts(y0, mv, y0 + a / 2 * X, y0 - a / 2 * X))
    # no unit at all: kept offsets only. Its least is what the offsets alone reach.
    t0 = y0.clone().requires_grad_(True)
    o = torch.optim.LBFGS([t0], lr=1, max_iter=500, line_search_fn="strong_wolfe", tolerance_grad=1e-11, tolerance_change=1e-15)

    def cl():
        o.zero_grad()
        v = kl_pts(y0, mv, t0, t0)
        v.backward()
        return v
    o.step(cl)
    none = float(cl().detach())
    fit = free_fit(y0, mv, 2 if smoke else 6, 200 if smoke else 600)
    ok = fit >= lo * (1 - 1e-6) and build <= up * (1 + 1e-6)
    print(f"{tag} q={q} t={t} n={n} a={a} E={len(e)} U={U} lower={lo:.2e} fit={fit:.5f} build={build:.5f} upper={up:.4f} no_unit={none:.4f} no_unit/fit={none / fit:.2f} ok={ok}", flush=True)
    return U, len(e), fit, ok


fails = 0
for a in (1.0, 4.0):
    for t in (1, 2):
        rows = []
        cases = [("C6", *cycle(6)), ("C5", *cycle(5)), ("C7", *cycle(7)), ("K4", *complete(4)), ("K5", *complete(5)), ("K3+K3", 6, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)])]
        if not smoke:
            for c in range(4):
                q = int(rng.integers(5, 8))
                ee = [p for p in itertools.combinations(range(q), 2) if rng.random() < .5] or [(0, 1)]
                cases.append((f"rnd{c}", q, ee))
        else:
            cases = cases[:3]
        for name, q, ee in cases:
            r = run(name, q, ee, t, a, .3 if t == 1 else .3)
            rows.append(r)
            fails += not r[3]
        Us, Es, Fs = map(np.array, zip(*[(x[0], x[1], x[2]) for x in rows]))
        rk = lambda x: np.argsort(np.argsort(x))
        print(f"a={a} t={t}: Spearman fit vs U {np.corrcoef(rk(Us), rk(Fs))[0, 1]:.2f}; fit vs E {np.corrcoef(rk(Es), rk(Fs))[0, 1]:.2f}", flush=True)
# holes M = sig sig^T
for name, q, ee in (("C5", *cycle(5)), ("C7", *cycle(7)), ("K4", *complete(4))):
    r = run("holes-signed " + name, q, ee, 2, 1.0, .3, hole=True)
    fails += not r[3]
print("fails", fails)
