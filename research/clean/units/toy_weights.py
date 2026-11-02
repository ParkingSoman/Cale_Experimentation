# serves: TU22 (UNIT-BRANCH.md)
"""Free boxes when every build re-fits its readers' bias AND their weights on kept units (the logits' weights on the
two gates). Same networks as toy_bias.py. Lab evidence (Adam from many starts)."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import toy_shelter as ts  # noqa: E402
from toy_district import make as make_old  # noqa: E402

torch.set_default_dtype(torch.float64)


def nets():
    X, Z, Wo = make_old("gated")
    yN = torch.relu(Z) @ Wo.T
    yield "orig", X, X, Z[:, :8], Wo[:, :8], yN, torch.ones(len(X)), {"S": [0, 1], "T": [0, 1]}, torch.relu(Z[:, 8:10])
    for kind in ("sheltered", "q4", "pose"):
        X, H1, Z2, Wo, yN, w, inb = ts.make(kind, False)
        yield kind, X, H1, Z2, Wo, yN, w, inb, H1[:, :2]


def fit(H1, Z2, Wo, yN, w, G, sets, ks, starts=16, steps=2500):
    best = float("inf")
    for st in range(starts):
        gen = torch.Generator().manual_seed(100 * st + 3)
        params = [(S, inb, (torch.randn(len(inb), k, generator=gen) * (1 + st % 3)).requires_grad_(),
                   (torch.randn(k, generator=gen) * (1 + st % 3)).requires_grad_(),
                   torch.randn(Wo.shape[0], k, generator=gen).requires_grad_()) for (S, inb), k in zip(sets, ks)]
        bias = torch.zeros(Wo.shape[0]).requires_grad_()
        dG = torch.zeros(Wo.shape[0], G.shape[1]).requires_grad_()  # change of the readers' weights on the kept gates
        opt = torch.optim.Adam([t for P in params for t in P[2:]] + [bias, dG], lr=.03)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for it in range(steps):
            loss = ts.kl(yN, ts.build(yN, H1, Z2, Wo, params) + bias + G @ dG.T, w)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        best = min(best, loss.item())
    return best


out = {}
for name, X, H1, Z2, Wo, yN, w, inb, G in nets():
    S, T = list(range(4)), list(range(4, 8))
    res = {}
    for kS, kT in ((1, 1), (2, 2), (1, 2)):
        dS = fit(H1, Z2, Wo, yN, w, G, [(S, inb["S"])], [kS]); dT = fit(H1, Z2, Wo, yN, w, G, [(T, inb["T"])], [kT])
        dST = fit(H1, Z2, Wo, yN, w, G, [(S, inb["S"]), (T, inb["T"])], [kS, kT])
        res[f"{kS}{kT}"] = dict(DS=dS, DT=dT, DST=dST, I=dST - dS - dT)
        print(name, kS, kT, {k: round(v, 5) for k, v in res[f"{kS}{kT}"].items()}, flush=True)
    out[name] = res
json.dump(out, open(os.path.join(os.path.dirname(__file__), "toy_weights.json"), "w"), indent=1)
