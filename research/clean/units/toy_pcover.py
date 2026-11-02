# serves: Remark B.3, TU5 (UNIT-BRANCH.md)
"""Price of independence with boxes that are covering ON P only: free class (b) units relu(m z_S + beta), any signs,
constrained to be off at the points of P outside supp(S) (penalty, then checked). Gated network of toy_district.py."""
import json
import os
import sys

import torch


from toy_district import make, kl, fit as _fit  # noqa: E402  (reuses the network and the free fit)

torch.manual_seed(0)


def fit_pcov(Z, Wo, yN, sets, ks, starts=16, steps=3000, lam=200.):
    best, bestp = float("inf"), None
    for st in range(starts):
        params = []
        for S, k in zip(sets, ks):
            out_mask = ~(Z[:, S] > 0).any(1)
            params.append((S, out_mask, (torch.randn(len(S), k) * (1 + st % 3)).requires_grad_(),
                           (torch.randn(k) * (1 + st % 3)).requires_grad_(), torch.randn(Wo.shape[0], k).requires_grad_()))
        opt = torch.optim.Adam([t for P in params for t in P[2:]], lr=.03)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for it in range(steps):
            y, pen = yN.clone(), 0.
            for S, om, m, be, V in params:
                pre = Z[:, S] @ m + be
                y = y - torch.relu(Z[:, S]) @ Wo[:, S].T + torch.relu(pre) @ V.T
                pen = pen + (torch.relu(pre[om]) ** 2).sum()
            loss = kl(yN, y); tot = loss + lam * pen
            opt.zero_grad(); tot.backward(); opt.step(); sch.step()
        # hard check: zero out any leak by clipping, then recompute exactly
        with torch.no_grad():
            y, leak = yN.clone(), 0.
            for S, om, m, be, V in params:
                pre = Z[:, S] @ m + be
                leak = max(leak, float(torch.relu(pre[om]).max()) if om.any() else 0.)
                h = torch.relu(pre); h[om] = 0.
                y = y - torch.relu(Z[:, S]) @ Wo[:, S].T + h @ V.T
            L = kl(yN, y).item()
        if L < best:
            best, bestp = L, dict(leak=leak)
    return best, bestp


X, Z, Wo = make("gated")
yN = torch.relu(Z) @ Wo.T
S, T = [0, 1, 2, 3], [4, 5, 6, 7]
out = {}
for kS, kT in ((1, 1), (2, 2)):
    dS, iS = fit_pcov(Z, Wo, yN, [S], [kS]); dT, iT = fit_pcov(Z, Wo, yN, [T], [kT])
    out[f"pcov_{kS}{kT}"] = dict(DS=dS, DT=dT, leaks=[iS["leak"], iT["leak"]])
    print(kS, kT, out[f"pcov_{kS}{kT}"], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "toy_pcover.json"), "w"), indent=1)
