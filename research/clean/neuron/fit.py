"""Scores for picture A (the parent absorbs). Reads results/absorb_cache.npz from absorb.py. CPU, float64.

For each parent, recipe and goal, on held-out reading photos:
  drop        children removed, readout unchanged
  closed      parent's readout column and bias changed by a squared-logit fit (no training; label OPT-D2)
  kl_opt      same two vectors, at the KL optimum on the fit photos (convex; started from zero, closed, random)
  whole       all of fc and the bias refit on KL (goal "all" only: too many numbers for the small goal)
"""
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


def kl_rows(yN, yB):
    lpN = torch.log_softmax(yN, 1)
    lpB = torch.log_softmax(yB, 1)
    return (lpN.exp() * (lpN - lpB)).sum(1)


def lbfgs(params, loss_fn, iters=300):
    opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, history_size=50, tolerance_grad=1e-10,
                            tolerance_change=1e-14, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        L = loss_fn()
        L.backward()
        return L
    opt.step(closure)
    return float(loss_fn())


def main():
    d = np.load(os.path.join(OUT, "absorb_cache.npz"))
    meta = json.load(open(os.path.join(OUT, "absorb_edges.json")))
    W = torch.tensor(d["fcW"]).double()
    bias = torch.tensor(d["fcb"]).double()
    lab, fit = d["lab"], d["fit"]
    gN = torch.tensor(d["gN"]).double()
    yN_all = gN @ W.T + bias
    parents = [int(p) for p in meta["parents"]]
    res = []
    for pi, p in enumerate(parents):
        par = d["par"][:, pi]
        cls_mean = np.array([par[(lab == c) & fit].mean() for c in range(1000)])
        own = np.argsort(-cls_mean)[:20]
        goals = {"all": np.ones(len(lab), bool), "own20": np.isin(lab, own)}
        for rec in ("map", "edge"):
            gD = torch.tensor(d[f"gD_{p}_{rec}"]).double()
            yD_all = gD @ W.T + bias
            for gname, gmask in goals.items():
                F = torch.tensor(np.where(gmask & fit)[0])
                T = torch.tensor(np.where(gmask & ~fit)[0])
                yN, yD, hp = yN_all[F], yD_all[F], gD[F, p]
                row = dict(parent=p, recipe=rec, children=meta["recipes"][f"{p}_{rec}"], goal=gname,
                           n_fit=len(F), n_test=len(T))
                row["drop"] = float(kl_rows(yN_all[T], yD_all[T]).mean())

                # closed form: centred logit gap regressed on [1, g_p]
                gap = (yN - yD)
                gap = gap - gap.mean(1, keepdim=True)
                hm = hp.mean()
                b = ((hp - hm)[:, None] * (gap - gap.mean(0))).mean(0) / ((hp - hm) ** 2).mean().clamp(min=1e-12)
                a = gap.mean(0) - b * hm

                def test_kl(a_, b_):
                    return float(kl_rows(yN_all[T], yD_all[T] + a_ + b_ * gD[T, p][:, None]).mean())
                row["closed"] = test_kl(a, b)
                row["closed_fit"] = float(kl_rows(yN, yD + a + b * hp[:, None]).mean())

                # KL optimum of the same form, three starts
                starts = {"zero": (torch.zeros(1000), torch.zeros(1000)), "closed": (a.clone(), b.clone()),
                          "random": (0.1 * torch.randn(1000), 0.1 * torch.randn(1000))}
                kl = {}
                for sname, (a0, b0) in starts.items():
                    A = a0.clone().requires_grad_(True)
                    B = b0.clone().requires_grad_(True)
                    with torch.enable_grad():
                        fl = lbfgs([A, B], lambda: kl_rows(yN, yD + A + B * hp[:, None]).mean())
                    kl[sname] = dict(fit=fl, test=test_kl(A.detach(), B.detach()))
                row["kl_opt"] = kl
                best = min(kl.values(), key=lambda v: v["fit"])
                row["kl_opt_test"] = best["test"]
                row["ratio_closed_over_klopt"] = row["closed"] / max(best["test"], 1e-12)

                if gname == "all":
                    W2 = W.clone().requires_grad_(True)
                    b2 = bias.clone().requires_grad_(True)
                    gF = gD[F]
                    with torch.enable_grad():
                        lbfgs([W2, b2], lambda: kl_rows(yN, gF @ W2.T + b2).mean(), iters=200)
                    row["whole"] = float(kl_rows(yN_all[T], gD[T] @ W2.detach().T + b2.detach()).mean())
                res.append(row)
                print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items() if k != "kl_opt"}),
                      flush=True)
    json.dump(res, open(os.path.join(OUT, "absorb_scores.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
