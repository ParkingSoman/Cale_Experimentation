# serves: DQ16
"""Part C. Is the weight pattern of a real table a product shape outside each row's near classes? Class-row tables of ResNet-18 and ResNet-50, cache/map (validation crops, 20 per class, not N's training photos).
Estimator, fixed before the run. Table: as in pc_real_cert.py, float64. Near classes of a row: its own class plus its family at J = 5, 10, 20 Ward clusters of N's class weight vectors, as in pc_real_minors.py; a measure of form, not a map. Product fit: log sbar[c, i] = alpha_c + beta_i on entries outside the near classes, by alternating weighted means, weights sbar. Reported at each J: (1) residual on strangers, unweighted and s-weighted: share of entries within a factor 3, quantiles; (2) residual of own class and of relatives above the product shape: median; (3) the share of the k = 14 table loss on own class, relatives, strangers, in the entrywise form s phi(log(q / s)), phi(e) = e^e - 1 - e, which is >= 0 and sums to the KL when q and s are normalised, on the saved fit fit_{net}_table_14.npz; (4) of the stranger loss, the share on entries within a factor 3 of the product shape; (5) eps_prod of DQ16 without eps_const. Then (6) the closed span test: V = top k right singular vectors of the centred log table with columns in the product metric, then a convex fit of t and c at fixed V; its loss against the saved fit, at k = 14, 38. An upper value, ONE-BUILD.
Prediction, written before the run: strangers fit the product shape within a factor 3 on most of the s-weighted mass; own class and relatives sit above it by e^5 or more; most of the loss sits on the strangers, and most of that where the product shape holds. Span test: under 1.5 times the fit.
Usage: pc_real_weights.py [--smoke]"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)
ITERS = 30 if a.smoke else 1500


def fit(ps, fn, it):
    o = torch.optim.LBFGS(ps, lr=1, max_iter=it, history_size=30, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-13)

    def cl():
        o.zero_grad()
        v = fn()
        v.backward()
        for p in ps:
            p.grad = p.grad.contiguous()
        return v
    o.step(cl)
    return float(fn().detach())


def product_fit(L, S, mask, iters=60):
    """alpha + beta on entries where mask is True, weights S, alternating weighted means."""
    w = S * mask
    alpha = torch.zeros(L.shape[0])
    beta = torch.zeros(L.shape[1])
    for _ in range(iters):
        alpha = (w * (L - beta[None, :])).sum(1) / w.sum(1)
        beta = (w * (L - alpha[:, None])).sum(0) / w.sum(0).clamp_min(1e-300)
    return alpha, beta


out = []
for net in ("resnet18", "resnet50"):
    Y = np.load(os.path.join(HERE, f"cache/map/{net}_y.npy"), mmap_mode="r")
    meta = np.load(os.path.join(HERE, f"cache/map/{net}_meta.npz"))
    n = Y.shape[0]
    Yt = torch.from_numpy(np.asarray(Y[:n]).astype(np.float64))
    lab = torch.from_numpy(meta["labels"][:n].astype(np.int64))
    W = meta["W"].astype(np.float64)
    S = torch.softmax(Yt, 1)
    ids = torch.unique(lab)
    sb = torch.stack([S[lab == c].mean(0) for c in ids]).clamp_min(1e-12)
    sb = sb / sb.sum(1, keepdim=True)
    K, C = sb.shape
    L = sb.log()
    Z = linkage(W, "ward")
    fitz = np.load(os.path.join(HERE, f"cache/map/fit_{net}_table_14.npz"))
    V14, c14, t14 = (torch.from_numpy(fitz[x].astype(np.float64)) for x in ("V", "c", "t"))
    q = torch.softmax(t14 + c14 @ V14.T, 1)
    e = (q / sb).log()
    ent = sb * (e.exp() - 1 - e) / K
    kl14 = float(ent.sum())
    for J in (5, 10, 20):
        fam = torch.from_numpy(fcluster(Z, J, "maxclust").astype(np.int64))
        idc = ids.long()
        own = torch.zeros(K, C, dtype=torch.bool)
        own[torch.arange(K), idc] = True
        rel = fam[idc][:, None] == fam[None, :]
        rel = rel & ~own
        str_ = ~(own | rel)
        alpha, beta = product_fit(L, sb, str_)
        res = L - alpha[:, None] - beta[None, :]
        rs = res[str_]
        ws = sb[str_]
        within = rs.abs() <= np.log(3)
        row = dict(net=net, J=J, K=K,
                   stranger_share_entries_within_factor3=float(within.double().mean()),
                   stranger_share_mass_within_factor3=float(ws[within].sum() / ws.sum()),
                   stranger_res_q=[float(torch.quantile(rs[torch.randperm(len(rs))[:200000]], x)) for x in (.05, .5, .95)],
                   own_above_median=float(res[own].median()),
                   rel_above_median=float(res[rel].median()),
                   rel_above_q=[float(torch.quantile(res[rel][torch.randperm(int(rel.sum()))[:200000]], x)) for x in (.05, .95)],
                   eps_prod_tail=float((sb[str_] * rs ** 2).sum() / K),
                   stranger_mass=float(sb[str_].sum() / K),
                   kl14=kl14, loss_share_own=float(ent[own].sum() / ent.sum()), loss_share_rel=float(ent[rel].sum() / ent.sum()), loss_share_str=float(ent[str_].sum() / ent.sum()), stranger_loss_within_factor3=float(ent[str_][within].sum() / ent[str_].sum()))
        out.append(row)
        print(row, flush=True)
    # (6) the closed span test: product metric from all entries
    alpha, beta = product_fit(L, sb, torch.ones_like(sb, dtype=torch.bool))
    cw = (beta - beta.max()).exp()
    cw = cw / cw.sum()
    D = L - (L * cw[None, :]).sum(1, keepdim=True)
    D = D - D.mean(0, keepdim=True)
    U, Sv, Vh = torch.linalg.svd(D * cw.sqrt()[None, :], full_matrices=False)
    for k in ((14,) if a.smoke else (14, 38)):
        Vc = torch.linalg.qr(Vh[:k].T / cw.sqrt()[:, None])[0]
        c = (D @ Vc).clone().requires_grad_(True)
        t = (L.mean(0)).clone().requires_grad_(True)
        v = fit([c, t], lambda: (sb * (L - torch.log_softmax(t + c @ Vc.T, 1))).sum(1).mean(), ITERS)
        f = np.load(os.path.join(HERE, f"cache/map/fit_{net}_table_{k}.npz"))
        qf = torch.log_softmax(torch.from_numpy(f["t"].astype(np.float64)) + torch.from_numpy(f["c"].astype(np.float64)) @ torch.from_numpy(f["V"].astype(np.float64)).T, 1)
        kf = float((sb * (L - qf)).sum(1).mean())
        row = dict(net=net, k=k, closed_span_loss=v, saved_fit=kf, ratio=v / kf)
        out.append(row)
        print(row, flush=True)
json.dump(out, open(os.path.join(HERE, "results_pc_real_weights_smoke.json" if a.smoke else "results_pc_real_weights.json"), "w"), indent=1)
