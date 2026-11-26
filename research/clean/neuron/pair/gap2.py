"""Gap, part 2: does training move each kept neuron's FIRING toward "own output + absorbed share of dropped outputs"?

Target for kept neuron n (train rows): T_n = g_n + sum_d B[n, d] g_d, B = regression of each dropped neuron's output
on all kept outputs (as in gap.py). On test rows:
  corr(g_new_n - g_old_n, T_n - g_old_n)  against B rows shuffled among kept neurons.
Then a no-training-of-the-network rule ("local refit"): each kept neuron, on its own, fits its read and bias so that
GELU(r . u + b) matches T_n (squared error, full-batch Adam, same steps for all, start = its own read). Then re-solve
the writes once. Score R^2 against the build (.675) and the trained layer (.709).
Variants: T with B from all dropped neurons; T with each dropped neuron given only to its top-1 absorber.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from cache import DEV
from distill import D


def main():
    torch.set_grad_enabled(False)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
    ck = torch.load(os.path.join(D, "ft_all.pt"), map_location="cpu", weights_only=False)
    st, sel, W1, b1 = ck["state"], ck["sel"], ck["W1"], ck["b1"]
    Utr = t(np.load(os.path.join(D, "U_train.npy"))[::6])
    Ytr = t(np.load(os.path.join(D, "Y_train.npy"))[::6])
    Ute = t(np.load(os.path.join(D, "U_test.npy"))[::2])
    Yte = t(np.load(os.path.join(D, "Y_test.npy"))[::2])
    mask = torch.ones(4096, dtype=torch.bool)
    mask[sel] = False
    drop = torch.where(mask)[0]
    Gtr = F.gelu(Utr @ W1.T + b1)
    Gk = torch.cat([Gtr[:, sel], torch.ones(len(Gtr), 1)], 1).double()
    A = Gk.T @ Gk
    A += 1e-3 * A.diagonal().mean() * torch.eye(len(A), dtype=torch.float64)
    B = torch.linalg.solve(A, Gk.T @ Gtr[:, drop].double())[:-1].float()   # (1024, 3072)
    top1 = torch.zeros_like(B)
    j = B.argmax(0)
    top1[j, torch.arange(B.shape[1])] = B[j, torch.arange(B.shape[1])].clamp_min(0)
    res = {}
    Gte = F.gelu(Ute @ W1.T + b1)
    g_old = Gte[:, sel]
    g_new = F.gelu(Ute @ st["l1.weight"].T + st["l1.bias"])
    perm = torch.randperm(1024, generator=torch.Generator().manual_seed(0))

    def colcorr(a, b):
        a = a - a.mean(0)
        b = b - b.mean(0)
        return (a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0) + 1e-12)
    for name, Bm in (("all", B), ("top1", top1)):
        T_te = g_old + Gte[:, drop] @ Bm.T
        res[f"corr_change_vs_target_{name}"] = float(colcorr(g_new - g_old, T_te - g_old).median())
        res[f"corr_change_vs_target_{name}_shuffled"] = float(colcorr(g_new - g_old, (Gte[:, drop] @ Bm.T)[:, perm]).median())
        res[f"corr_new_vs_target_{name}"] = float(colcorr(g_new, T_te).median())
        res[f"corr_old_vs_target_{name}"] = float(colcorr(g_old, T_te).median())
    print(json.dumps(res, indent=1), flush=True)

    yv = (Yte - Yte.mean(0)).pow(2).sum(1).mean()

    def resolve_r2(R, bias):
        g = F.gelu(Utr @ R.T + bias)
        X = torch.cat([g, torch.ones(len(g), 1)], 1).double()
        A = X.T @ X
        A += 1e-3 * A.diagonal().mean() * torch.eye(len(A), dtype=torch.float64)
        W = torch.linalg.solve(A, X.T @ Ytr.double()).float()
        gt = F.gelu(Ute @ R.T + bias)
        e = Yte - torch.cat([gt, torch.ones(len(gt), 1)], 1) @ W
        e = e - e.mean(0)
        return float(1 - e.pow(2).sum(1).mean() / yv)

    res["build"] = resolve_r2(W1[sel], b1[sel])
    res["trained_reads"] = resolve_r2(st["l1.weight"], st["l1.bias"])
    for name, Bm in (("all", B), ("top1", top1)):
        T = (Gtr[:, sel] + Gtr[:, drop] @ Bm.T).to(DEV)
        U = Utr.to(DEV)
        R = W1[sel].clone().to(DEV).requires_grad_(True)
        bb = b1[sel].clone().to(DEV).requires_grad_(True)
        with torch.enable_grad():
            opt = torch.optim.Adam([R, bb], lr=1e-3)
            for step in range(400):
                idx = torch.randint(0, len(U), (8192,), device=DEV)
                loss = (F.gelu(U[idx] @ R.T + bb) - T[idx]).pow(2).mean(0).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
        res[f"local_refit_{name}"] = resolve_r2(R.detach().cpu(), bb.detach().cpu())
        print(name, res[f"local_refit_{name}"], flush=True)
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(D, "gap2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
