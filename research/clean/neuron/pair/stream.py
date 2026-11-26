"""Pair study: is each Small block-0 neuron a single read direction on the Base stream?

Base block-0 neurons read u = LayerNorm(dwconv(stream)) (1,024 numbers per position, after the LN's affine).
Neuron p's pre-activation is W1[p] . u + b1[p]. If a Small neuron s is a coarser version of p, the best linear read
of s from u gives a vector v_s, and v_s can be compared with p's read vector W1[p].

One pass over the photos of pos.py (2 fit + 2 test per class). Sums on fit and test rows give, for all 3,072 Small
block-0 neurons:
  R2_stream    s ~ a + v . u        (fit on fit rows, scored on test rows)
  cos(v_s, W1[p]) for the counterpart p (in the metric of u's fit covariance, so it compares what the two read)
  how v_s splits over Base read vectors: least squares of v_s on W1 rows (all 4,096 rows span u), and the share of
  ||v_s|| (covariance metric) carried by p alone and by the top 5 / 20 / 100 rows.
"""
import json
import os
import time

import numpy as np
import torch

from cache import DEV, OUT, hook_stage4, load, net
from pos import photos

torch.set_grad_enabled(False)


def main():
    ph = net.Photos("read")
    tr, te = photos(ph)
    ms, mb = load("small"), load("base")
    st_s = {}
    hook_stage4(ms, st_s)
    U = {}
    mb.features[7][0].block[2].register_forward_hook(lambda m, i, o: U.__setitem__("u", o))
    D = 1024
    z = lambda *s: torch.zeros(*s, dtype=torch.float64)
    A = {k: dict(n=0, su=z(D), suu=z(D, D), ss=z(3072), sss=z(3072), ssu=z(3072, D)) for k in ("fit", "test")}
    allix = np.sort(np.concatenate([tr, te]))
    tes = set(te.tolist())
    t0 = time.time()
    for k in range(0, len(allix), 50):
        ii = allix[k:k + 50]
        x = ph.tensor(ii).to(DEV)
        ms(x), mb(x)
        u = U["u"].reshape(len(ii), 49, D).float()
        s = st_s[0].reshape(len(ii), 49, 3072).float()
        for name, mask in (("fit", [i not in tes for i in ii]), ("test", [i in tes for i in ii])):
            mk = torch.tensor(mask, device=DEV)
            if not mk.any():
                continue
            uu, ss = u[mk].reshape(-1, D), s[mk].reshape(-1, 3072)
            a = A[name]
            a["n"] += uu.shape[0]
            a["su"] += uu.sum(0).cpu().double()
            a["suu"] += (uu.T @ uu).cpu().double()
            a["ss"] += ss.sum(0).cpu().double()
            a["sss"] += (ss * ss).sum(0).cpu().double()
            a["ssu"] += (ss.T @ uu).cpu().double()
        if (k // 50) % 20 == 0:
            print(f"  {k + len(ii)}/{len(allix)} {time.time() - t0:.0f}s", flush=True)

    f, t = A["fit"], A["test"]
    n = f["n"]
    mu, ms_ = f["su"] / n, f["ss"] / n
    Cuu = f["suu"] / n - mu[:, None] * mu[None, :]
    Csu = f["ssu"] / n - ms_[:, None] * mu[None, :]
    V = torch.linalg.solve(Cuu + 1e-6 * torch.eye(D, dtype=torch.float64), Csu.T).T  # (3072, D)
    a0 = ms_ - V @ mu
    # test SSE from test sums: sum (s - a0 - V u)^2
    nt = t["n"]
    sse = (t["sss"] - 2 * a0 * t["ss"] - 2 * (V * t["ssu"]).sum(1) + nt * a0 ** 2
           + 2 * a0 * (V @ t["su"]) + ((V @ t["suu"]) * V).sum(1))
    vt = t["sss"] / nt - (t["ss"] / nt) ** 2
    r2 = (1 - sse / nt / vt).numpy()

    W1 = mb.features[7][0].block[3].weight.detach().cpu().double()  # (4096, D)
    L = torch.linalg.cholesky(Cuu + 1e-6 * torch.eye(D, dtype=torch.float64))  # covariance metric
    Vm = V @ L          # rows: v_s in the metric
    Wm = W1 @ L
    Rf = np.load(os.path.join(OUT, "tools_peakR_b0.npy")).astype(np.float32)
    Rw = np.load(os.path.join(OUT, "writesR_b0.npy")).astype(np.float32)
    agree = np.where(Rf.argmax(1) == Rw.argmax(1))[0]
    pmap = Rf.argmax(1)
    cos_p, cos_best, best_is_p, share_p, share_top = [], [], [], [], {5: [], 20: [], 100: []}
    Wn = Wm / Wm.norm(dim=1, keepdim=True)
    for s in agree:
        v = Vm[s]
        c = (Wn @ v / v.norm()).numpy()
        cos_p.append(c[pmap[s]])
        cos_best.append(c.max())
        best_is_p.append(int(c.argmax() == pmap[s]))
        # greedy: how much of v is explained by p alone, and by the top-k rows (least squares on those rows)
        order = np.argsort(-np.abs(c))
        for kk in (5, 20, 100):
            Wk = Wm[order[:kk]]
            coef = torch.linalg.lstsq(Wk.T, v[:, None]).solution
            share_top[kk].append(float(1 - ((Wk.T @ coef)[:, 0] - v).norm() ** 2 / v.norm() ** 2))
        share_p.append(float(c[pmap[s]] ** 2))
    res = dict(r2_stream_median_all=float(np.median(r2)), r2_stream_median_agree=float(np.median(r2[agree])),
               r2_stream_q10_q90=[float(np.quantile(r2, .1)), float(np.quantile(r2, .9))],
               cos_v_p_median=float(np.median(cos_p)), cos_v_best_row_median=float(np.median(cos_best)),
               best_row_is_p=float(np.mean(best_is_p)), share_p_median=float(np.median(share_p)),
               share_top_median={k: float(np.median(v)) for k, v in share_top.items()}, n_agree=int(len(agree)))
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(OUT, "stream.json"), "w"), indent=1)
    np.save(os.path.join(OUT, "stream_V.npy"), V.float().numpy())
    np.savez(os.path.join(OUT, "stream_cov.npz"), Cuu=Cuu.numpy(), mu=mu.numpy(), W1=W1.numpy(),
             W1s=ms.features[7][0].block[3].weight.detach().cpu().double().numpy(), agree=agree, pmap=pmap)
    np.save(os.path.join(OUT, "stream_r2.npy"), r2.astype(np.float32))


if __name__ == "__main__":
    main()
