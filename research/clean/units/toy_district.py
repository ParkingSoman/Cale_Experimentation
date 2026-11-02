# serves: DU2, TU5, TU7 (UNIT-BRANCH.md)
"""Step 7 toy: districts with gates. Interaction of least boxes for S = V's members and T = A's members,
free class (b) boxes and covering boxes (m >= 0, beta <= 0), on a gated network and on a control where one V member
and one A member fire together. Least builds by Adam from many starts (lab evidence)."""
import json
import math
import os
import sys

import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
deg = math.pi / 180


def make(kind):
    if kind == "gated":
        arcV, arcA = (-50, 50), (130, 230)
        thV = [-37.5, -12.5, 12.5, 37.5]; thA = [142.5, 167.5, 192.5, 217.5]; r0V = r0A = math.cos(20 * deg)
    else:  # control: arcs touch, one V member reaches into A's arc, no gates
        arcV, arcA = (-50, 50), (55, 155)
        thV = [-37.5, -12.5, 12.5, 37.5]; thA = [67.5, 92.5, 117.5, 142.5]; r0V = math.cos(35 * deg); r0A = math.cos(20 * deg)
    g = torch.Generator().manual_seed(1)
    ang = torch.cat([torch.linspace(*arcV, 24), torch.linspace(*arcA, 24)]) * deg
    rad = 1 + .05 * (torch.rand(48, generator=g) - .5)
    X = torch.stack([rad * torch.cos(ang), rad * torch.sin(ang)], 1)
    rows, bias, writes = [], [], []
    for th, r0, cls in [(t, r0V, i // 2) for i, t in enumerate(thV)] + [(t, r0A, 2 + i // 2) for i, t in enumerate(thA)]:
        s = 1 / (1 - r0)
        rows.append([s * math.cos(th * deg), s * math.sin(th * deg)]); bias.append(-s * r0)
        w = [0.] * 4; w[cls] = 3.; w[cls ^ 1] = -1.; writes.append(w)
    if kind == "gated":
        rows += [[4., 0.], [-4., 0.]]; bias += [-1.2, -1.2]
        writes += [[1.5, 1.5, -1.5, -1.5], [-1.5, -1.5, 1.5, 1.5]]
    W = torch.tensor(rows); b = torch.tensor(bias); Wo = torch.tensor(writes).T
    Z = X @ W.T + b
    return X, Z, Wo


def kl(yN, yB):
    a, b = torch.log_softmax(yN, 1), torch.log_softmax(yB, 1)
    return (a.exp() * (a - b)).sum(1).mean()


def build(yN, Z, Wo, params, cover):
    y = yN.clone()
    for S, m, be, V in params:
        mm = torch.nn.functional.softplus(m) if cover else m
        bb = -torch.nn.functional.softplus(be) if cover else be
        y = y - torch.relu(Z[:, S]) @ Wo[:, S].T + torch.relu(Z[:, S] @ mm + bb) @ V.T
    return y


def fit(Z, Wo, yN, sets, ks, cover, starts=16, steps=2500):
    """least D over boxes for the given sets with budgets ks; returns (loss, params)."""
    best, bestp = float("inf"), None
    for st in range(starts):
        params = []
        for S, k in zip(sets, ks):
            m = (torch.randn(len(S), k) * (1 + st % 3)).requires_grad_()
            be = (torch.randn(k) * (1 + st % 3)).requires_grad_()
            V = (torch.randn(Wo.shape[0], k)).requires_grad_()
            params.append((S, m, be, V))
        opt = torch.optim.Adam([t for P in params for t in P[1:]], lr=.03)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for it in range(steps):
            loss = kl(yN, build(yN, Z, Wo, params, cover))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if loss.item() < best:
            best, bestp = loss.item(), [(S, m.detach(), be.detach(), V.detach()) for S, m, be, V in params]
    return best, bestp


def main():
    BUDGETS = [(1, 1)] if "--smoke" in sys.argv else [(1, 1), (2, 2), (1, 2)]
    out = {}
    for kind in ("gated", "control"):
        X, Z, Wo = make(kind)
        yN = torch.relu(Z) @ Wo.T
        S, T = [0, 1, 2, 3], [4, 5, 6, 7]
        on = Z > 0
        cofire = bool((on[:, S].any(1) & on[:, T].any(1)).any())
        res = dict(cofire_S_T=cofire, n_cofire_points=int((on[:, S].any(1) & on[:, T].any(1)).sum()))
        for cover in (False, True):
            tag = "cov" if cover else "free"
            for kS, kT in BUDGETS:
                dS, pS = fit(Z, Wo, yN, [S], [kS], cover)
                dT, pT = fit(Z, Wo, yN, [T], [kT], cover)
                dST, pST = fit(Z, Wo, yN, [S, T], [kS, kT], cover)
                # the joint's boxes, one at a time (upper bounds on the single optima, optimiser-free identity for TU5)
                eS = kl(yN, build(yN, Z, Wo, pST[:1], cover)).item()
                eT = kl(yN, build(yN, Z, Wo, pST[1:], cover)).item()
                # the single optima used together
                ePair = kl(yN, build(yN, Z, Wo, pS + pT, cover)).item()
                sS, sT = min(dS, eS), min(dT, eT)
                r = dict(DS=dS, DT=dT, DST=dST, joint_boxes_alone=[eS, eT], singles_together=ePair,
                         identity_gap=dST - eS - eT, I_est=min(dST, ePair) - sS - sT,
                         rel=(min(dST, ePair) - sS - sT) / max(sS + sT, 1e-12))
                res[f"{tag}_{kS}{kT}"] = r
                print(kind, tag, kS, kT, {k: (round(v, 5) if isinstance(v, float) else [round(u, 5) for u in v]) for k, v in r.items()}, flush=True)
        out[kind] = res
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "toy_district.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
