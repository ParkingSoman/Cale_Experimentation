"""Gap, part 3: redundancy. Do the value-ranked survivors overlap, and does training spread them apart?

1. Redundancy of the kept outputs (test rows): median |correlation| between kept neurons, and the effective number of
   directions (participation ratio of the eigenvalues of their covariance), build vs trained.
2. Greedy selection (no training): pick survivors that best reduce the error of rebuilding y, given the ones already
   picked (batch orthogonal matching pursuit on covariances: 64 per step), then re-solve writes. Compare with
   value ranking (.675) and trained (.709). Also greedy on the top 2,048 by value only.
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
    st, sel, W1, b1 = ck["state"], ck["sel"], ck["W1"], ck["b1"]
    Utr = t(np.load(os.path.join(D, "U_train.npy"))[::6])
    Ytr = t(np.load(os.path.join(D, "Y_train.npy"))[::6])
    Ute = t(np.load(os.path.join(D, "U_test.npy"))[::2])
    Yte = t(np.load(os.path.join(D, "Y_test.npy"))[::2])
    res = {}
    g_old = F.gelu(Ute @ W1[sel].T + b1[sel])
    g_new = F.gelu(Ute @ st["l1.weight"].T + st["l1.bias"])
    for name, g in (("build", g_old), ("trained", g_new)):
        Cg = torch.cov(g.T.double())
        sd = Cg.diagonal().sqrt()
        R = Cg / sd[:, None] / sd[None, :]
        off = R[~torch.eye(1024, dtype=torch.bool)].abs()
        ev = torch.linalg.eigvalsh(R)
        res[f"{name}_median_abs_corr"] = float(off.median())
        res[f"{name}_q99_abs_corr"] = float(off.quantile(.99)) if off.numel() < 16_000_000 else float(off[::7].quantile(.99))
        res[f"{name}_effective_dims"] = float(ev.sum() ** 2 / (ev ** 2).sum())
    print(json.dumps(res, indent=1), flush=True)

    # greedy selection on covariances (train rows)
    G = F.gelu(Utr @ W1.T + b1).double()
    G = G - G.mean(0)
    Yc = (Ytr - Ytr.mean(0)).double()
    K = G.T @ G / len(G)                 # (4096, 4096)
    Q = G.T @ Yc / len(G)                # (4096, 1024)
    del G
    yv = (Yte - Yte.mean(0)).pow(2).sum(1).mean()

    def resolve_r2(idx):
        g = F.gelu(Utr @ W1[idx].T + b1[idx])
        X = torch.cat([g, torch.ones(len(g), 1)], 1).double()
        A = X.T @ X
        A += 1e-3 * A.diagonal().mean() * torch.eye(len(A), dtype=torch.float64)
        W = torch.linalg.solve(A, X.T @ Ytr.double()).float()
        gt = F.gelu(Ute @ W1[idx].T + b1[idx])
        e = Yte - torch.cat([gt, torch.ones(len(gt), 1)], 1) @ W
        e = e - e.mean(0)
        return float(1 - e.pow(2).sum(1).mean() / yv)

    def greedy(cands, k=1024, batch=64):
        S = []
        ridge = 1e-6 * K.diagonal().mean()
        while len(S) < k:
            if S:
                KSS = K[np.ix_(S, S)] + ridge * torch.eye(len(S), dtype=torch.float64)
                KcS = K[:, S]
                Z = torch.linalg.solve(KSS, KcS.T)               # (|S|, 4096)
                Wt = torch.linalg.solve(KSS, Q[S])               # (|S|, 1024)
                corr = Q - KcS @ Wt                              # covariance of each candidate with the residual
                denom = K.diagonal() - (KcS * Z.T).sum(1)
            else:
                corr, denom = Q, K.diagonal().clone()
            gain = corr.pow(2).sum(1) / denom.clamp_min(1e-9)
            gain[S] = -1
            mask = torch.full((4096,), -1.0, dtype=torch.float64)
            mask[cands] = gain[cands]
            mask[S] = -1
            take = torch.argsort(-mask)[:min(batch, k - len(S))]
            S += take.tolist()
        return torch.tensor(S)

    value_rank = torch.argsort(-(F.gelu(Utr @ W1.T + b1).std(0) * 1.0))  # placeholder (not used)
    res["value_ranked_resolved"] = resolve_r2(sel)
    S_all = greedy(torch.arange(4096))
    res["greedy_all_resolved"] = resolve_r2(S_all)
    res["greedy_overlap_with_value_top1024"] = float(np.isin(S_all.numpy(), sel.numpy()).mean())
    print(json.dumps(res, indent=1), flush=True)
    torch.save(dict(S=S_all), os.path.join(D, "greedy_S.pt"))
    json.dump(res, open(os.path.join(D, "gap3.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
