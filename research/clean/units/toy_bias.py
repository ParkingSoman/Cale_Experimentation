# serves: TU17, TU18 (UNIT-BRANCH.md)
"""Free boxes when every build may re-fit its readers' biases (here the readers are the logits, so: a free logit
bias in every build, single and joint). Networks: the original circle toy (members read the raw input), and the
two-layer sheltered, q4 and pose toys of toy_shelter.py. Lab evidence (Adam from many starts)."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import toy_shelter as ts  # noqa: E402
from toy_district import make as make_old  # noqa: E402

torch.set_default_dtype(torch.float64)


def make_orig():
    X, Z, Wo = make_old("gated")
    H1 = X  # the members read the raw input; a box of either set reads it too
    Z2 = Z[:, :8]
    yN = torch.relu(Z) @ Wo.T
    return X, H1, Z2, Wo[:, :8], yN, torch.ones(len(X)), {"S": [0, 1], "T": [0, 1]}


def fit_bias(H1, Z2, Wo, yN, w, sets, ks, starts=16, steps=2500):
    best, bestp = float("inf"), None
    for st in range(starts):
        gen = torch.Generator().manual_seed(100 * st + 3)
        params = [(S, inb, (torch.randn(len(inb), k, generator=gen) * (1 + st % 3)).requires_grad_(),
                   (torch.randn(k, generator=gen) * (1 + st % 3)).requires_grad_(),
                   torch.randn(Wo.shape[0], k, generator=gen).requires_grad_()) for (S, inb), k in zip(sets, ks)]
        bias = torch.zeros(Wo.shape[0]).requires_grad_()
        opt = torch.optim.Adam([t for P in params for t in P[2:]] + [bias], lr=.03)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for it in range(steps):
            loss = ts.kl(yN, ts.build(yN, H1, Z2, Wo, params) + bias, w)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if loss.item() < best:
            best, bestp = loss.item(), ([(S, inb, m.detach(), be.detach(), V.detach()) for S, inb, m, be, V in params], bias.detach())
    return best, bestp


def main():
    out = {}
    nets = [("orig", lambda: make_orig()), ("sheltered", lambda: ts.make("sheltered", False)),
            ("q4", lambda: ts.make("q4", False)), ("pose", lambda: ts.make("pose", False))]
    for name, mk in nets:
        X, H1, Z2, Wo, yN, w, inb = mk()
        S, T = list(range(4)), list(range(4, 8))
        onS = (Z2[:, S] > 0).any(1); onT = (Z2[:, T] > 0).any(1)
        res = {}
        for kS, kT in ((1, 1), (2, 2), (1, 2)):
            dS, _ = fit_bias(H1, Z2, Wo, yN, w, [(S, inb["S"])], [kS])
            dT, _ = fit_bias(H1, Z2, Wo, yN, w, [(T, inb["T"])], [kT])
            dST, (P, b) = fit_bias(H1, Z2, Wo, yN, w, [(S, inb["S"]), (T, inb["T"])], [kS, kT])
            r = dict(DS=dS, DT=dT, DST=dST); r["I"] = dST - dS - dT
            units = []
            for bx, (Sset, ib, m, be, V) in zip(("S", "T"), P):
                h = torch.relu(H1[:, ib] @ m + be)
                for j in range(h.shape[1]):
                    on = h[:, j] > 0
                    units.append(dict(box=bx, on_S=int((on & onS).sum()), on_T=int((on & onT).sum()),
                                      below=int((on & (X[:, 1] < 0)).sum()), above=int((on & (X[:, 1] > 0)).sum()),
                                      top_read=ib[int(m[:, j].abs().argmax())], write=[round(float(v), 2) for v in V[:, j]]))
            r["units"] = units
            res[f"{kS}{kT}"] = r
            print(name, kS, kT, {k: round(v, 4) for k, v in r.items() if k != "units"}, flush=True)
            for u in units:
                print("    ", u, flush=True)
        out[name] = res
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "toy_bias.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
