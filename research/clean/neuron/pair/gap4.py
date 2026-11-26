"""Gap, part 4: is the gap a LINEAR part of the dropped neurons' output that thresholded survivors cannot make?

1. Pre-activation scale: std of each kept neuron's pre-activation, trained / build; share of rows above 0.
2. Least squares (train rows), scored on test rows:
     linear only:        y ~ V u
     build:              y ~ W g_sel
     build + linear:     y ~ W g_sel + V u        (one solve; no training)
     greedy + linear:    same with the greedy survivors (gap3)
     trained reads + linear
3. How much of the dropped neurons' output (sum_d w_d g_d) is linear in u, and how much is linear in the kept outputs.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from cache import load
from distill import D

torch.set_grad_enabled(False)


def main():
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
    ck = torch.load(os.path.join(D, "ft_all.pt"), map_location="cpu", weights_only=False)
    st, sel, W1, b1 = ck["state"], ck["sel"], ck["W1"], ck["b1"]
    S_g = torch.load(os.path.join(D, "greedy_S.pt"), weights_only=False)["S"]
    Utr = t(np.load(os.path.join(D, "U_train.npy"))[::6])
    Ytr = t(np.load(os.path.join(D, "Y_train.npy"))[::6])
    Ute = t(np.load(os.path.join(D, "U_test.npy"))[::2])
    Yte = t(np.load(os.path.join(D, "Y_test.npy"))[::2])
    res = {}
    p_old = Ute @ W1[sel].T + b1[sel]
    p_new = Ute @ st["l1.weight"].T + st["l1.bias"]
    res["preact_std_ratio_median"] = float((p_new.std(0) / p_old.std(0)).median())
    res["on_share_build_median"] = float((p_old > 0).float().mean(0).median())
    res["on_share_trained_median"] = float((p_new > 0).float().mean(0).median())
    yv = lambda Y: (Y - Y.mean(0)).pow(2).sum(1).mean()

    def fit(Xtr, Xte, Ytr_=Ytr, Yte_=Yte):
        X = torch.cat([Xtr, torch.ones(len(Xtr), 1)], 1).double()
        A = X.T @ X
        A += 1e-4 * A.diagonal().mean() * torch.eye(len(A), dtype=torch.float64)
        W = torch.linalg.solve(A, X.T @ Ytr_.double()).float()
        e = Yte_ - torch.cat([Xte, torch.ones(len(Xte), 1)], 1) @ W
        e = e - e.mean(0)
        return float(1 - e.pow(2).sum(1).mean() / yv(Yte_))

    g = lambda U, idx: F.gelu(U @ W1[idx].T + b1[idx])
    res["linear_only"] = fit(Utr, Ute)
    res["build"] = fit(g(Utr, sel), g(Ute, sel))
    res["build_plus_linear"] = fit(torch.cat([g(Utr, sel), Utr], 1), torch.cat([g(Ute, sel), Ute], 1))
    res["greedy"] = fit(g(Utr, S_g), g(Ute, S_g))
    res["greedy_plus_linear"] = fit(torch.cat([g(Utr, S_g), Utr], 1), torch.cat([g(Ute, S_g), Ute], 1))
    gn = lambda U: F.gelu(U @ st["l1.weight"].T + st["l1.bias"])
    res["trained_reads"] = fit(gn(Utr), gn(Ute))
    res["trained_reads_plus_linear"] = fit(torch.cat([gn(Utr), Utr], 1), torch.cat([gn(Ute), Ute], 1))
    # dropped neurons' output
    m = load("base")
    blk = m.features[7][0]
    ls = blk.layer_scale.detach().float().cpu().view(-1)
    Wb = ls[:, None] * blk.block[5].weight.detach().float().cpu()
    del m, blk
    mask = torch.ones(4096, dtype=torch.bool)
    mask[sel] = False
    drop = torch.where(mask)[0]
    Ydr_tr = g(Utr, drop) @ Wb[:, drop].T
    Ydr_te = g(Ute, drop) @ Wb[:, drop].T
    res["dropped_output_share_of_y_var"] = float(yv(Ydr_te) / yv(Yte))
    res["dropped_output_linear_in_u"] = fit(Utr, Ute, Ydr_tr, Ydr_te)
    res["dropped_output_linear_in_kept"] = fit(g(Utr, sel), g(Ute, sel), Ydr_tr, Ydr_te)
    res["dropped_output_linear_in_kept_and_u"] = fit(torch.cat([g(Utr, sel), Utr], 1), torch.cat([g(Ute, sel), Ute], 1),
                                                     Ydr_tr, Ydr_te)
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(D, "gap4.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
