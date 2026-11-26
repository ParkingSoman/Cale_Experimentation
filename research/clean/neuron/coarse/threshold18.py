"""N11 M1 follow-up: is the threshold the missing piece of the rule's coarse neuron?
The plain sum (multilevel18.py) fires at 9% of positions against B's 42%: a sum of weakly correlated sub-threshold
inputs needs several members on at once (an AND), while a coarse neuron must fire when any member fires (an OR).
Variants per family f (pre_f = sum of members' pre), each at one level (block 0) and two levels (blocks 0 and 1):
  any : pre_f + t_f, t_f set so P(on) = P(any member on) on calibration positions (from A alone)
  rB  : pre_f + t_f, t_f set so P(on) = .42 (B k128's measured median block-0 rate: a reading of B)
  or  : g_f = max over members of g_c (an OR of the members; not a single neuron, a check)
Write for all: sum_c a_c W2[:, c], a_c = E[g_c] / E[g_f]; one mean map per output channel keeps the mean output.
Families: co-firing ("cof") and shared feeders ("rec"). Scored on 19,000 held + test photos.
Saved: results/coarse/r18/threshold.json
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import r18
import multilevel18 as ML

torch.set_grad_enabled(False)
K, BL, A = ML.K, ML.BL, ML.A
RATE_B = 0.42


def build(b, lab, mode):
    bl = BL[b]
    M = torch.zeros(K, 512, device=r18.DEV)
    M[torch.as_tensor(lab, device=r18.DEV), torch.arange(512, device=r18.DEV)] = 1
    t = torch.zeros(K, device=r18.DEV)
    if mode in ("any", "rB"):
        S, anyon, n = [], torch.zeros(K, device=r18.DEV), 0
        for h in ML.cal_inputs(b):
            pre = bl.bn1(bl.conv1(h))
            pf = torch.einsum("kc,nchw->nkhw", M, pre)
            S.append(pf[:, :, ::2, ::2].permute(1, 0, 2, 3).reshape(K, -1).cpu())
            anyon += torch.einsum("kc,nchw->nkhw", M, (pre > 0).float()).gt(0).float().mean((2, 3)).sum(0); n += len(h)
        S = torch.cat(S, 1)
        rate = (anyon / n).cpu() if mode == "any" else torch.full((K,), RATE_B)
        q = torch.stack([torch.quantile(S[k], float(1 - rate[k])) for k in range(K)])
        t = -q.to(r18.DEV)

    def gf_of(pre, g):
        if mode == "or":
            return torch.stack([g[:, M[k] > 0].amax(1) for k in range(K)], 1)
        return torch.relu(torch.einsum("kc,nchw->nkhw", M, pre) + t.view(1, K, 1, 1))

    Eg = torch.zeros(512, 7, 7, device=r18.DEV); Egf = torch.zeros(K, 7, 7, device=r18.DEV); on = torch.zeros(K, device=r18.DEV); n = 0
    for h in ML.cal_inputs(b):
        pre = bl.bn1(bl.conv1(h)); g = torch.relu(pre); gf = gf_of(pre, g)
        Eg += g.sum(0); Egf += gf.sum(0); on += (gf > 0).float().mean((2, 3)).sum(0); n += len(h)
    Eg, Egf = Eg / n, Egf / n
    a = Eg.mean((1, 2)) / Egf.mean((1, 2))[torch.as_tensor(lab, device=r18.DEV)].clamp_min(1e-6)
    W2f = torch.einsum("ocij,kc->okij", bl.conv2.weight, M * a[None, :])
    corr = F.conv2d(Eg[None], bl.conv2.weight, padding=1)[0] - F.conv2d(Egf[None], W2f, padding=1)[0]
    return dict(gf_of=gf_of, W2f=W2f, corr=corr, rate=float((on / n).median()))


def fwd(b, h, P):
    bl = BL[b]
    pre = bl.bn1(bl.conv1(h)); g = torch.relu(pre)
    return ML.finish(bl, F.conv2d(P["gf_of"](pre, g), P["W2f"], padding=1) + P["corr"][None], h)


def main():
    head = lambda h: r18.net.head(A, h)
    P = {(f, m): [build(0, ML.FAM[f][0], m), build(1, ML.FAM[f][1], m)] for f in ("cof", "rec") for m in ("any", "rB", "or")}
    print("built", {f"{f}_{m}": [round(p["rate"], 3) for p in v] for (f, m), v in P.items()}, flush=True)
    V = {}
    for (f, m), (p0, p1) in P.items():
        V[f"L0_{m}_{f}"] = (lambda p0: lambda x: head(BL[1](fwd(0, x, p0))))(p0)
        V[f"L01_{m}_{f}"] = (lambda p0, p1: lambda x: head(fwd(1, fwd(0, x, p0), p1)))(p0, p1)
    lab_all = r18.net.Photos("read").lab
    rows = {k: {m: [] for m in ("kl", "hit", "cos", "spread")} for k in V}
    for s in ("held", "test"):
        c = r18.Cache(s)
        for a in range(0, len(c), 250):
            rr = np.arange(a, min(a + 250, len(c)))
            x, _ = c.get(rr); lab = lab_all[c.ix[rr]]
            zA = r18.run4(A, x); p = F.softmax(zA, 1); lp = F.log_softmax(zA, 1); ca = zA - zA.mean(1, keepdim=True)
            for k, fn in V.items():
                z = fn(x); lq = F.log_softmax(z, 1); cz = z - z.mean(1, keepdim=True)
                rows[k]["kl"].append((p * (lp - lq)).sum(1).cpu().numpy())
                rows[k]["hit"].append(z.argmax(1).cpu().numpy() == lab)
                rows[k]["cos"].append(F.cosine_similarity(cz - ca, ca, 1).cpu().numpy())
                rows[k]["spread"].append((cz.std(1) / ca.std(1)).cpu().numpy())
        print("scored", s, flush=True)
    res = {"rates": {f"{f}_{m}": [p["rate"] for p in v] for (f, m), v in P.items()}}
    for k in V:
        d = {m: np.concatenate(v) for m, v in rows[k].items()}
        res[k] = dict(kl=float(d["kl"].mean()), top1=float(d["hit"].mean()), cos_median=float(np.median(d["cos"])),
                      spread_median=float(np.median(d["spread"])))
    json.dump(res, open(os.path.join(r18.OUT, "threshold.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
