"""Pair study, step 1 tool check: which matching tool finds one-to-one counterparts?

The first tool (correlation over all photo positions, pre-GELU, same block) gave a mean best match of 0.44 on 500
photos. The essay says counterparts exist across ConvNeXt sizes, so a weak number blames the tool. This script
compares tools on the same photos (fit half only):

  pre       correlation over positions, pre-GELU             (same block, and every Small block x Base block)
  gelu      correlation over positions, after GELU           (same block)
  peak      correlation of the per-photo maximum             (same block)
  top       overlap of the top 1% photos by per-photo maximum (same block): |A and B| / |A|

For each tool: the best Base match per Small neuron, and the one-to-one (Hungarian) assignment strength.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from cache import DEV, NB, OUT, hook_stage4, load, net, split

torch.set_grad_enabled(False)


class Acc:
    def __init__(self, dx, dy):
        z = lambda *s: torch.zeros(*s, dtype=torch.float64)
        self.n, self.sx, self.sy, self.sxx, self.syy, self.sxy = 0, z(dx), z(dy), z(dx), z(dy), z(dx, dy)

    def add(self, X, Y):
        self.n += X.shape[0]
        self.sx += X.sum(0).cpu().double()
        self.sy += Y.sum(0).cpu().double()
        self.sxx += (X * X).sum(0).cpu().double()
        self.syy += (Y * Y).sum(0).cpu().double()
        self.sxy += (X.T @ Y).cpu().double()

    def corr(self):
        n = self.n
        mx, my = self.sx / n, self.sy / n
        vx = (self.sxx / n - mx ** 2).clamp_min(1e-12)
        vy = (self.syy / n - my ** 2).clamp_min(1e-12)
        return ((self.sxy / n - mx[:, None] * my[None, :]) / vx.sqrt()[:, None] / vy.sqrt()[None, :]).float().numpy()


def summarize(R):
    best = R.max(1)
    r, c = linear_sum_assignment(-R)
    hung = R[r, c]
    return dict(best_mean=float(best.mean()), best_median=float(np.median(best)),
                best_over={t: float((best > t).mean()) for t in (0.4, 0.6, 0.8)},
                hung_mean=float(hung.mean()), hung_over={t: float((hung > t).mean()) for t in (0.4, 0.6, 0.8)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=50)
    a = ap.parse_args()
    ph = net.Photos("read")
    fit = split(ph.lab)
    pool = np.where(fit)[0]
    idx = np.sort(np.random.default_rng(1).choice(pool, a.n, replace=False))
    ms, mb = load("small"), load("base")
    st_s, st_b = {}, {}
    hook_stage4(ms, st_s)
    hook_stage4(mb, st_b)
    Ds, Db = 3072, 4096
    pre = {(i, j): Acc(Ds, Db) for i in range(NB) for j in range(NB)}
    gel = {i: Acc(Ds, Db) for i in range(NB)}
    pk_s = np.zeros((len(idx), NB, Ds), np.float32)
    pk_b = np.zeros((len(idx), NB, Db), np.float32)
    gelu = torch.nn.GELU()
    t0 = time.time()
    for k in range(0, len(idx), a.batch):
        ii = idx[k:k + a.batch]
        x = ph.tensor(ii).to(DEV)
        ms(x), mb(x)
        Xs = {b: st_s[b].reshape(-1, Ds).float() for b in range(NB)}
        Yb = {b: st_b[b].reshape(-1, Db).float() for b in range(NB)}
        for b in range(NB):
            pk_s[k:k + len(ii), b] = st_s[b].amax((1, 2)).cpu().numpy()
            pk_b[k:k + len(ii), b] = st_b[b].amax((1, 2)).cpu().numpy()
            gel[b].add(gelu(Xs[b]), gelu(Yb[b]))
            for j in range(NB):
                pre[(b, j)].add(Xs[b], Yb[j])
        if (k // a.batch) % 10 == 0:
            print(f"  {k + len(ii)}/{len(idx)}  {time.time() - t0:.0f}s", flush=True)

    res = {}
    for b in range(NB):
        res[f"pre b{b}-b{b}"] = summarize(pre[(b, b)].corr())
        res[f"gelu b{b}"] = summarize(gel[b].corr())
        # peak: correlation of per-photo maxima
        xs = (pk_s[:, b] - pk_s[:, b].mean(0)) / (pk_s[:, b].std(0) + 1e-6)
        yb = (pk_b[:, b] - pk_b[:, b].mean(0)) / (pk_b[:, b].std(0) + 1e-6)
        Rp = xs.T @ yb / len(idx)
        res[f"peak b{b}"] = summarize(Rp)
        # top: overlap of top 1% photos
        kq = max(int(0.01 * len(idx)), 5)
        As = np.zeros((len(idx), Ds), np.float32)
        Ab = np.zeros((len(idx), Db), np.float32)
        np.put_along_axis(As, np.argsort(-pk_s[:, b], 0)[:kq], 1, 0)
        np.put_along_axis(Ab, np.argsort(-pk_b[:, b], 0)[:kq], 1, 0)
        res[f"top b{b}"] = summarize(As.T @ Ab / kq)
        np.save(os.path.join(OUT, f"tools_peakR_b{b}.npy"), Rp.astype(np.float16))
        for j in range(NB):
            if j != b:
                R = pre[(b, j)].corr()
                res[f"pre b{b}-b{j}"] = dict(best_mean=float(R.max(1).mean()),
                                             best_over_06=float((R.max(1) > .6).mean()))
    for k_, v in res.items():
        print(f"{k_:12s} " + " ".join(f"{kk}={vv}" for kk, vv in v.items()), flush=True)
    json.dump(dict(n=len(idx), tools=res), open(os.path.join(OUT, f"tools_{a.n}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, f"tools_peaks_{a.n}.npz"), idx=idx, small=pk_s.astype(np.float16),
                        base=pk_b.astype(np.float16))
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
