# serves: TU17, TU18 (UNIT-BRANCH.md)
"""Is free 'help' a property of the network? Two-layer circle toys. Layer 1: gates gV, gA (also written straight to
the logits) and four narrow caps per district (silent outside their arc); optionally shared pose units up, down.
Layer 2: each district's four members read only their own district's caps (and, in the 'pose' network, one pose
unit). Free boxes read the set's in-boundary (layer-1 units). Mix: the two arcs, optionally plus background points
where no district is on. Least builds by Adam from many starts (lab evidence)."""
import json
import math
import os
import sys

import torch

torch.set_default_dtype(torch.float64)
deg = math.pi / 180


def make(kind, background):
    g = torch.Generator().manual_seed(1)
    ang = torch.cat([torch.linspace(-50, 50, 24), torch.linspace(130, 230, 24)])
    w = torch.ones(48)
    if background:
        ang = torch.cat([ang, torch.linspace(80, 100, 12), torch.linspace(260, 280, 12)])
        w = torch.cat([w, torch.full((24,), 2.)])  # equal mass: 48 arc points, 24 background points of weight 2
    ang = ang * deg
    rad = 1 + .05 * (torch.rand(len(ang), generator=g) - .5)
    X = torch.stack([rad * torch.cos(ang), rad * torch.sin(ang)], 1)
    thV = [-37.5, -12.5, 12.5, 37.5]; thA = [142.5, 167.5, 192.5, 217.5]; r0 = math.cos(20 * deg); s = 1 / (1 - r0)
    caps = [torch.relu(s * (X[:, 0] * math.cos(t * deg) + X[:, 1] * math.sin(t * deg)) - s * r0) for t in thV + thA]
    gV = torch.relu(4 * X[:, 0] - 1.2); gA = torch.relu(-4 * X[:, 0] - 1.2)
    H1 = [gV, gA] + caps  # 0 gV, 1 gA, 2..5 V caps, 6..9 A caps
    if kind.startswith("pose"):
        H1 += [torch.relu(3 * X[:, 1]), torch.relu(-3 * X[:, 1])]  # 10 up, 11 down (shared, not gated)
    H1 = torch.stack(H1, 1)
    nin = H1.shape[1]
    W2 = torch.zeros(8, nin); b2 = torch.zeros(8)
    lower = {0: True, 1: True, 2: False, 3: False, 4: False, 5: False, 6: True, 7: True}  # V1, V2 / A1 (upper), A2 (lower)
    for i in range(8):
        W2[i, 2 + i] = 1.
        if kind.startswith("pose"):
            W2[i, 11 if lower[i] else 10] = .3; b2[i] = -.9
        if kind.endswith("q4"):  # a gate in front: members read their district's gate, still off where their cap is 0
            W2[i, 0 if i < 4 else 1] = .2; b2[i] -= .56
    Z2 = H1 @ W2.T + b2
    Wo = torch.zeros(4, 8)
    for i in range(8):
        cls = i // 2; Wo[cls, i] = 3.; Wo[cls ^ 1, i] = -1.
    Wg = torch.tensor([[1.5, -1.5], [1.5, -1.5], [-1.5, 1.5], [-1.5, 1.5]])  # gates -> logits
    yN = torch.relu(Z2) @ Wo.T + H1[:, :2] @ Wg.T
    inb = {"S": [j for j in range(nin) if W2[:4, j].abs().sum() > 0], "T": [j for j in range(nin) if W2[4:, j].abs().sum() > 0]}
    return X, H1, Z2, Wo, yN, w, inb


def kl(yN, yB, w):
    a, b = torch.log_softmax(yN, 1), torch.log_softmax(yB, 1)
    return ((a.exp() * (a - b)).sum(1) * w).sum() / w.sum()


def build(yN, H1, Z2, Wo, params):
    y = yN.clone()
    for S, inb, m, be, V in params:
        y = y - torch.relu(Z2[:, S]) @ Wo[:, S].T + torch.relu(H1[:, inb] @ m + be) @ V.T
    return y


def fit(H1, Z2, Wo, yN, w, sets, ks, starts=16, steps=2500):
    best, bestp = float("inf"), None
    for st in range(starts):
        gen = torch.Generator().manual_seed(100 * st + 3)
        params = []
        for (S, inb), k in zip(sets, ks):
            params.append((S, inb, (torch.randn(len(inb), k, generator=gen) * (1 + st % 3)).requires_grad_(),
                           (torch.randn(k, generator=gen) * (1 + st % 3)).requires_grad_(),
                           torch.randn(Wo.shape[0], k, generator=gen).requires_grad_()))
        opt = torch.optim.Adam([t for P in params for t in P[2:]], lr=.03)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for it in range(steps):
            loss = kl(yN, build(yN, H1, Z2, Wo, params), w)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if loss.item() < best:
            best, bestp = loss.item(), [(S, inb, m.detach(), be.detach(), V.detach()) for S, inb, m, be, V in params]
    return best, bestp


def main():
    out = {}
    for kind, bg in (("sheltered", False), ("sheltered", True), ("pose", False), ("pose", True)):
        X, H1, Z2, Wo, yN, w, inb = make(kind, bg)
        S, T = list(range(4)), list(range(4, 8))
        onS = (Z2[:, S] > 0).any(1); onT = (Z2[:, T] > 0).any(1)
        inbS_var_on_T = float((H1[onT][:, inb["S"]].max(0).values - H1[onT][:, inb["S"]].min(0).values).max())
        res = dict(points_S=int(onS.sum()), points_T=int(onT.sum()), cofire=int((onS & onT).sum()),
                   S_inputs_max_range_on_T_points=round(inbS_var_on_T, 3))
        for kS, kT in ((1, 1), (2, 2), (1, 2)):
            dS, _ = fit(H1, Z2, Wo, yN, w, [(S, inb["S"])], [kS])
            dT, _ = fit(H1, Z2, Wo, yN, w, [(T, inb["T"])], [kT])
            dST, P = fit(H1, Z2, Wo, yN, w, [(S, inb["S"]), (T, inb["T"])], [kS, kT])
            eS = kl(yN, build(yN, H1, Z2, Wo, P[:1]), w).item(); eT = kl(yN, build(yN, H1, Z2, Wo, P[1:]), w).item()
            r = dict(DS=min(dS, eS), DT=min(dT, eT), DST=dST)
            r["I"] = r["DST"] - r["DS"] - r["DT"]
            # where do the joint's box units fire, and which input do they read most
            units = []
            for name, (Sset, ib, m, be, V) in zip(("S", "T"), P):
                h = torch.relu(H1[:, ib] @ m + be)
                for j in range(h.shape[1]):
                    on = h[:, j] > 0
                    top = ib[int(m[:, j].abs().argmax())]
                    units.append(dict(box=name, on_S_points=int((on & onS).sum()), on_T_points=int((on & onT).sum()),
                                      on_other_points=int((on & ~onS & ~onT).sum()), top_read_input=top,
                                      write=[round(float(v), 2) for v in V[:, j]]))
            r["joint_units"] = units
            res[f"{kS}{kT}"] = r
            print(kind, "bg" if bg else "arcs", kS, kT, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "joint_units"}, flush=True)
            for u in units:
                print("    ", u, flush=True)
        out[f"{kind}_{'bg' if bg else 'arcs'}"] = res
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "toy_shelter.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
