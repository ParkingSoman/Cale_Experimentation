# serves: TQ17
"""Part C. The realism row of TQ17 used the pairing by greatest confusion. The referee (and Theorem TQ17 part 5, proved after it) shows that at t = 0 a family table has its partner no likelier than a tail. So the pairing the theorem asks for is by LEAST confusion. Rerun the pole form with that pairing.
Estimator, fixed before the run: class-row tables of ResNet-18 and ResNet-50 (cache/map, one row per class, k = 500), and of the CIFAR-100 net of H-stat (its own training photos, k = 50). A pairing by least confusion, greedy on the sum of the two cross entries. Fit at the pole span, codes and offset free, L-BFGS 1500 iterations from 0; the distance of the residual from the projector pattern; the premise numbers: at the fit, the row-mean of m_c and of q'_c, and mu_fit. Also how much better the pole span is than the pairing by greatest confusion on the same table, with the free fit from a random start for reference.
A measure of form, not a map.
Prediction, written before the run: least confusion lowers the pole span's KL below greatest confusion, by under a factor of 3, and stays over 100 times the free fit. The essay says a unit keeps a close relative on its negative pole, so I expect real networks to show no sign of this pairing.
Usage: pc_real_pole.py [--smoke]"""
import argparse
import json
import os
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(6)
torch.set_default_dtype(torch.float64)
ITERS = 30 if a.smoke else 1500


def fit(params, fn):
    opt = torch.optim.LBFGS(params, lr=1, max_iter=ITERS, history_size=30, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-13)

    def cl():
        opt.zero_grad()
        v = fn()
        v.backward()
        for p in params:
            p.grad = p.grad.contiguous()
        return v
    opt.step(cl)
    return float(fn().detach())


def pairing(sb, least):
    K = len(sb)
    conf = sb + sb.T
    conf.fill_diagonal_(float("inf") if least else -1.0)
    pair = -np.ones(K, dtype=int)
    for f in torch.argsort(conf.flatten(), descending=not least).tolist():
        i, j = f // K, f % K
        if i != j and pair[i] < 0 and pair[j] < 0:
            pair[i], pair[j] = j, i
    return pair


def study(name, sb):
    K = len(sb)
    k = K // 2
    lb = sb.log()
    pi = torch.ones(K) / K
    row = dict(table=name, K=K, k=k)
    tab = lambda t, c, V: (pi * (sb * (lb - torch.log_softmax(t + c @ V.T, 1))).sum(1)).sum()
    for label, least in (("least", True), ("greatest", False)):
        t0 = time.time()
        pair = pairing(sb, least)
        Vp = torch.zeros(K, k)
        j = 0
        for c_ in range(K):
            if c_ < pair[c_]:
                Vp[c_, j], Vp[pair[c_], j] = 1, -1
                j += 1
        cp = torch.zeros(K, k).requires_grad_(True)
        tp = lb.mean(0).clone().requires_grad_(True)
        v = fit([cp, tp], lambda: tab(tp, cp, Vp))
        q = torch.softmax(tp + cp @ Vp.T, 1).detach()
        Ps = torch.zeros(K, K)
        Ps[torch.arange(K), torch.from_numpy(pair)] = 1
        Pi = .5 * (torch.eye(K) + Ps) - torch.ones(K, K) / K
        res = sb - q
        mu = float((res * Pi).sum() / (Pi * Pi).sum())
        ix = torch.arange(K)
        partner = sb[ix, torch.from_numpy(pair)]
        tail_mean = (1 - sb[ix, ix] - partner) / (K - 2)
        row[label] = dict(kl=v, residual_l1=float(res.abs().sum(1).mean()), l1_after_pattern=float((res - mu * Pi).abs().sum(1).mean()), mu_fit=mu, partner_mean=float(partner.mean()), share_partner_below_tail_mean=float((partner < tail_mean).double().mean()), sec=time.time() - t0)
        print(name, label, row[label], flush=True)
    torch.manual_seed(1)
    V = (torch.randn(K, k) * .3).requires_grad_(True)
    c = (torch.randn(K, k) * .3).requires_grad_(True)
    t = lb.mean(0).clone().requires_grad_(True)
    row["free"] = fit([V, c, t], lambda: tab(t, c, V))
    print(name, "free", row["free"], flush=True)
    return row


out = []
for net in ("resnet18", "resnet50"):
    Y = np.load(os.path.join(HERE, f"cache/map/{net}_y.npy"), mmap_mode="r")
    lab = torch.from_numpy(np.load(os.path.join(HERE, f"cache/map/{net}_meta.npz"))["labels"].astype(np.int64))
    S = torch.softmax(torch.from_numpy(np.asarray(Y).astype(np.float64)), 1)
    sb = torch.stack([S[lab == c].mean(0) for c in range(1000)]).clamp_min(1e-12)
    sb = sb / sb.sum(1, keepdim=True)
    out.append(study(net, sb))
    if a.smoke:
        break
dat = np.load(os.path.join(HERE, "cache/map/cifar_net.npz"))
s = torch.softmax(torch.tensor(dat["h"]).double() @ torch.tensor(dat["W"]).double().T + torch.tensor(dat["b"]).double(), 1)
lab = torch.tensor(dat["lab"]).long()
sb = torch.stack([s[lab == c].mean(0) for c in range(100)])
sb = sb / sb.sum(1, keepdim=True)
out.append(study("cifar", sb))
if not a.smoke:
    json.dump(out, open(os.path.join(HERE, "results_pc_real_pole.json"), "w"), indent=1)
print("done", flush=True)
