"""Build a coarse block-0 layer from Base's own neurons, with no training, and score it against distillation.

Rule:
  1. keep the k Base neurons with the largest value = std(output) x |write| (train rows)
  2. keep their reads (optionally shift each bias by delta x its pre-activation std: fire more broadly)
  3. re-solve their writes once: ridge least squares of Base's MLP output y on the kept outputs (train rows).
     Each kept neuron's new write = its own write + what it can carry of the dropped neurons' writes.
Scores: R^2 of y on test rows; end to end (k = 1024): top-1 and KL(Base || Base with the layer swapped).
Reference: the distilled layer trained from random at the same k (distill.py).
All math in torch (a numpy matmul after loading torch was killed by the system here).
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from cache import DEV, load
from distill import D, MLP, plug_eval


def main():
    torch.set_grad_enabled(False)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
    Utr = t(np.load(os.path.join(D, "U_train.npy"))[::6])
    Ytr = t(np.load(os.path.join(D, "Y_train.npy"))[::6])
    Ute = t(np.load(os.path.join(D, "U_test.npy"))[::2])
    Yte = t(np.load(os.path.join(D, "Y_test.npy"))[::2])
    m = load("base")
    blk = m.features[7][0]
    W1 = blk.block[3].weight.detach().float().cpu().clone()
    b1 = blk.block[3].bias.detach().float().cpu().clone()
    ls = blk.layer_scale.detach().float().cpu().view(-1).clone()
    Wb = ls[:, None] * blk.block[5].weight.detach().float().cpu()
    bb = ls * blk.block[5].bias.detach().float().cpu()
    del m, blk
    yc = Yte - Yte.mean(0)
    yv = yc.pow(2).sum(1).mean()

    def r2(Yh):
        e = Yte - Yh
        e = e - e.mean(0)
        return float(1 - e.pow(2).sum(1).mean() / yv)

    Gtr = F.gelu(Utr @ W1.T + b1)
    value = Gtr.std(0) * Wb.norm(dim=0)
    del Gtr
    print("value done", flush=True)
    out = {}
    for k in (1024, 512, 2048):
        sel = torch.argsort(-value)[:k]
        res = {}
        for delta in (0.0, 0.25, 0.5):
            pre_tr = Utr @ W1[sel].T + b1[sel]
            bshift = b1[sel] + delta * pre_tr.std(0)
            gtr = F.gelu(pre_tr + (bshift - b1[sel]))
            gte = F.gelu(Ute @ W1[sel].T + bshift)
            if delta == 0:
                res["writes_unchanged"] = r2(gte @ Wb[:, sel].T + bb)
            X = torch.cat([gtr, torch.ones(len(gtr), 1)], 1).double()
            A = X.T @ X
            A += 1e-3 * A.diagonal().mean() * torch.eye(len(A), dtype=torch.float64)
            Wn = torch.linalg.solve(A, X.T @ Ytr.double()).float()  # (k+1, 1024)
            res[f"writes_resolved_shift{delta}"] = r2(torch.cat([gte, torch.ones(len(gte), 1)], 1) @ Wn)
            if k == 1024 and delta == 0.0:
                mlp = MLP(k)
                mlp.l1.weight.copy_(W1[sel])
                mlp.l1.bias.copy_(bshift)
                mlp.l2.weight.copy_(Wn[:k].T)
                mlp.l2.bias.copy_(Wn[k])
                res["end_to_end_resolved"] = plug_eval(mlp.to(DEV).eval(), k, -1)
                mlp0 = MLP(k)
                mlp0.l1.weight.copy_(W1[sel])
                mlp0.l1.bias.copy_(b1[sel])
                mlp0.l2.weight.copy_(Wb[:, sel])
                mlp0.l2.bias.copy_(bb)
                res["end_to_end_unchanged"] = plug_eval(mlp0.to(DEV).eval(), k, -1)
            print(k, delta, {a: b for a, b in res.items() if not isinstance(b, dict)}, flush=True)
        if k == 1024:
            res["distilled_trained"] = [json.load(open(os.path.join(D, f"eval_k1024_s{s}.json"))) for s in (0, 1)]
        out[f"k{k}"] = res
        json.dump(out, open(os.path.join(D, "absorb_build.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
