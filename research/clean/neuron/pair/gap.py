"""What does training change beyond the no-training build? (k = 1024, block 0, stage 4 of convnext_base)

Build: keep the top-value Base neurons (reads r_n, biases), re-solve writes by least squares.
Trained: ft_all.pt (the build, then 10 epochs of training on all weights).

1. Where the read changes: |delta r_n| / |r_n| in the covariance metric of u; how concentrated the change is.
2. Absorption shares: B[n, d] = coefficient of kept neuron n when the dropped neuron d's output is regressed on all kept
   outputs (train rows). Prediction: delta r_n points toward sum_d B[n, d] * rt_d, rt_d = r_d / std(pre_d)
   ("fire where the neurons you absorbed fire"). Score: metric cosine with delta r_n; controls: B rows shuffled
   among kept neurons; the plain sum of all dropped reads.
3. A no-training rule from that: r_n' = r_n + lam * std(pre_n) * sum_d B[n, d] rt_d, then re-solve writes. Test R^2.
4. Where the gap lives: per test row, the error drop from build to trained against the output energy of the
   dropped neurons on that row.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from distill import D

torch.set_grad_enabled(False)


def main():
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
    ck = torch.load(os.path.join(D, "ft_all.pt"), map_location="cpu", weights_only=False)
    st, sel, W1, b1, Wn = ck["state"], ck["sel"], ck["W1"], ck["b1"], ck["Wn"]
    Utr = t(np.load(os.path.join(D, "U_train.npy"))[::12])
    Ytr = t(np.load(os.path.join(D, "Y_train.npy"))[::12])
    Ute = t(np.load(os.path.join(D, "U_test.npy"))[::2])
    Yte = t(np.load(os.path.join(D, "Y_test.npy"))[::2])
    mask = torch.ones(4096, dtype=torch.bool)
    mask[sel] = False
    drop = torch.where(mask)[0]
    C = torch.cov(Utr[::2].T.double())
    L = torch.linalg.cholesky(C + 1e-6 * torch.eye(1024, dtype=torch.float64))
    M = lambda r: r.double() @ L                       # rows -> metric coordinates
    r_old, r_new = W1[sel], st["l1.weight"]
    dM = M(r_new - r_old)
    oM = M(r_old)
    rel = dM.norm(dim=1) / oM.norm(dim=1)
    res = dict(read_change_rel_median=float(rel.median()), read_change_rel_q90=float(rel.quantile(.9)),
               bias_change_median=float((st["l1.bias"] - b1[sel]).abs().median()),
               share_of_change_in_top10pct_neurons=float(torch.sort(dM.norm(dim=1) ** 2, descending=True).values[:102].sum()
                                                         / (dM.norm(dim=1) ** 2).sum()))
    # absorption shares
    pre = Utr @ W1.T + b1
    G = F.gelu(pre)
    Gk = torch.cat([G[:, sel], torch.ones(len(G), 1)], 1).double()
    A = Gk.T @ Gk
    A += 1e-3 * A.diagonal().mean() * torch.eye(len(A), dtype=torch.float64)
    B = torch.linalg.solve(A, Gk.T @ G[:, drop].double())[:-1]     # (1024 kept, 3072 dropped)
    sd_pre = pre.std(0)
    rt = W1[drop] / sd_pre[drop, None]                              # dropped reads, unit pre-activation std
    pred = B.float() @ rt                                            # (1024, 1024)
    pM = M(pred)
    cosM = lambda a, b: (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)
    perm = torch.randperm(1024, generator=torch.Generator().manual_seed(0))
    res["cos_delta_vs_absorbed_reads_median"] = float(cosM(dM, pM).median())
    res["cos_delta_vs_shuffled_median"] = float(cosM(dM, pM[perm]).median())
    allsum = M(rt.sum(0, keepdim=True)).expand_as(dM)
    res["cos_delta_vs_all_dropped_median"] = float(cosM(dM, allsum).median())
    # also: does delta point away from the neuron's own read? (shrink)
    res["cos_delta_vs_own_read_median"] = float(cosM(dM, oM).median())

    # no-training rule
    yv = (Yte - Yte.mean(0)).pow(2).sum(1).mean()

    def build_r2(R, bias):
        g = F.gelu(Utr @ R.T + bias)
        X = torch.cat([g, torch.ones(len(g), 1)], 1).double()
        A = X.T @ X
        A += 1e-3 * A.diagonal().mean() * torch.eye(len(A), dtype=torch.float64)
        W = torch.linalg.solve(A, X.T @ Ytr.double()).float()
        gt = F.gelu(Ute @ R.T + bias)
        e = Yte - torch.cat([gt, torch.ones(len(gt), 1)], 1) @ W
        e = e - e.mean(0)
        return float(1 - e.pow(2).sum(1).mean() / yv)

    res["rule"] = {"lam0": build_r2(r_old, b1[sel])}
    for lam in (0.1, 0.3, 0.6, 1.0):
        Rn = r_old + lam * sd_pre[sel, None] * pred
        res["rule"][f"lam{lam}"] = build_r2(Rn, b1[sel])
        res["rule"][f"lam{lam}_shuffled"] = build_r2(r_old + lam * sd_pre[sel, None] * pred[perm], b1[sel])
    res["trained_reads_resolved_writes"] = build_r2(r_new, st["l1.bias"])

    # where the gap lives (test rows)
    def out(R, b, W2, c):
        return F.gelu(Ute @ R.T + b) @ W2.T + c
    Gte = Ute @ W1.T + b1
    yb = torch.cat([F.gelu(Gte[:, sel]), torch.ones(len(Ute), 1)], 1) @ Wn
    yt = out(r_new, st["l1.bias"], st["l2.weight"], st["l2.bias"])
    eb = (Yte - yb).pow(2).sum(1)
    et = (Yte - yt).pow(2).sum(1)
    from cache import load
    m = load("base")
    blk = m.features[7][0]
    ls = blk.layer_scale.detach().float().cpu().view(-1)
    Wb = (ls[:, None] * blk.block[5].weight.detach().float().cpu())
    dropE = (F.gelu(Gte[:, drop]) @ Wb[:, drop].T).pow(2).sum(1)    # energy of the dropped neurons' output
    q = torch.quantile(dropE, torch.tensor([.5, .9, .99]))
    bins = {"bottom50": dropE <= q[0], "50-90": (dropE > q[0]) & (dropE <= q[1]), "90-99": (dropE > q[1]) & (dropE <= q[2]),
            "top1": dropE > q[2]}
    tot = float((eb - et).sum())
    res["gap_share_by_dropped_energy"] = {k: float((eb - et)[v].sum() / tot) for k, v in bins.items()}
    res["gap_total_share_of_build_error"] = float(tot / eb.sum())
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(D, "gap.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
