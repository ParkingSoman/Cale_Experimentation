# serves: TU12, TU13 (UNIT-BRANCH.md)
"""Output-faithful boxes (Definition DU6): each box unit is either off at the points of P outside its set's support
(covering on P) or a constant unit (Lemma TU12, generic case). Interaction on three instances, with N's logit bias
fixed and with a free logit bias in every build. Least builds by Adam from many starts (lab evidence)."""
import itertools
import json
import math
import os
import sys
from fractions import Fraction as Fr

import torch

sys.path.insert(0, os.path.dirname(__file__))
from toy_district import make, kl  # noqa: E402

torch.set_default_dtype(torch.float64)


def d2(yN, yB):
    return ((yN - yB) ** 2).sum(1).mean()


def fit(Z, Wo, yN, lossf, sets, ks, freebias, starts=8, steps=2500, lam=200.):
    """least loss over output-faithful boxes: for each set, every split of its budget into covering-on-P units and
    constant units is tried."""
    best = float("inf")
    splits = [range(k + 1) for k in ks]
    for js in itertools.product(*splits):
        for st in range(starts):
            g = torch.Generator().manual_seed(1000 * st + 7)
            params, ten = [], []
            for S, k, j in zip(sets, ks, js):
                om = ~(Z[:, S] > 0).any(1)
                m = (torch.randn(len(S), k - j, generator=g) * (1 + st % 3)).requires_grad_()
                be = (torch.randn(k - j, generator=g) * (1 + st % 3)).requires_grad_()
                V = torch.randn(Wo.shape[0], k - j, generator=g).requires_grad_()
                C = torch.zeros(Wo.shape[0], j).requires_grad_()  # constant units: value 1, write C
                params.append((S, om, m, be, V, C)); ten += [m, be, V, C]
            bias = torch.zeros(Wo.shape[0]).requires_grad_()
            if freebias:
                ten.append(bias)
            opt = torch.optim.Adam(ten, lr=.03)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

            def run(clip):
                y, pen = yN.clone() + bias, 0.
                for S, om, m, be, V, C in params:
                    pre = Z[:, S] @ m + be
                    h = torch.relu(pre)
                    if clip:
                        h = h.clone(); h[om] = 0.
                    pen = pen + (torch.relu(pre[om]) ** 2).sum()
                    y = y - torch.relu(Z[:, S]) @ Wo[:, S].T + h @ V.T + C.sum(1)
                return y, pen
            for it in range(steps):
                y, pen = run(False)
                tot = lossf(yN, y) + lam * pen
                opt.zero_grad(); tot.backward(); opt.step(); sch.step()
            with torch.no_grad():
                L = lossf(yN, run(True)[0]).item()
            best = min(best, L)
    return best


def interaction(Z, Wo, yN, lossf, S, T, kS, kT, freebias):
    dS = fit(Z, Wo, yN, lossf, [S], [kS], freebias)
    dT = fit(Z, Wo, yN, lossf, [T], [kT], freebias)
    dST = fit(Z, Wo, yN, lossf, [S, T], [kS, kT], freebias)
    return dict(DS=dS, DT=dT, DST=dST, I=dST - dS - dT)


def t61():
    def circle_pt(i):
        return (Fr(1 - i * i, 1 + i * i), Fr(2 * i, 1 + i * i))
    xs = [circle_pt(1), circle_pt(2)]
    eta = 1 - max(a[0] * b[0] + a[1] * b[1] for a, b in itertools.permutations(xs, 2))
    r = 1 - eta / 8
    c = 2 / (1 - r)
    units = [(0, 0), (0, 1), (1, 0), (1, 1)]  # A, B, C, D

    def vertex(u):
        j, i = units[u]; x = [Fr(0)] * 4; x[2 * j], x[2 * j + 1] = xs[i]; return x
    bad = [(0, 1), (2, 3), (0, 2), (1, 3)]
    pts = [vertex(u) for u in range(4)] + [[(p + q) / 2 for p, q in zip(vertex(a), vertex(b))] for a, b in bad]
    Z = torch.tensor([[float(xs[i][0] * x[2 * j] + xs[i][1] * x[2 * j + 1] - r) for j, i in units] for x in pts])
    Wo = torch.full((1, 4), float(c))
    return Z, Wo


def main():
    out = {}
    # |x| on {-2,-1,1,2}, D2 on the single logit gap
    xv = torch.tensor([-2., -1., 1., 2.])
    Z = torch.stack([xv, -xv], 1); Wo = torch.ones(1, 2); yN = torch.relu(Z) @ Wo.T
    for fb in (False, True):
        out[f"abs_freebias{int(fb)}"] = interaction(Z, Wo, yN, d2, [0], [1], 2, 0, fb)
        print("abs", fb, out[f"abs_freebias{int(fb)}"], flush=True)
    # circle toy with gates (KL)
    X, Z, Wo = make("gated"); yN = torch.relu(Z) @ Wo.T
    for fb in (False, True):
        for kS, kT in ((1, 1), (2, 2), (1, 2)):
            key = f"toy_{kS}{kT}_freebias{int(fb)}"
            out[key] = interaction(Z, Wo, yN, kl, [0, 1, 2, 3], [4, 5, 6, 7], kS, kT, fb)
            print(key, out[key], flush=True)
    # T61 instance (D2 on the gap)
    Z, Wo = t61(); yN = torch.relu(Z) @ Wo.T
    for fb in (False, True):
        for kS, kT in ((1, 1), (1, 2)):
            key = f"t61_{kS}{kT}_freebias{int(fb)}"
            out[key] = interaction(Z, Wo, yN, d2, [0, 2], [1, 3], kS, kT, fb)
            print(key, out[key], flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "toy_faithful.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
