"""N6 step 3b: do the 30 blocks' errors add up or cancel, for the build B and the trained T?

For X in (B, T) and each block g: Base with only block g's MLP replaced by X's narrow MLP. d_g = logits - Base logits
(per photo, centred over classes). On the 1,500 test photos:
  single     sum over g of KL(Base || Base with X's block g)
  linear     KL(Base || Base logits + sum_g d_g)
  actual     KL(Base || X with all 30 blocks)  (from the ledger: B .370, T .131)
  alignment  per photo |sum_g d_g|^2 / sum_g |d_g|^2 in the softmax metric e^T (diag p - p p^T) e; median and mean
  pairs      mean cosine (softmax metric) between d_g and d_h, adjacent and far pairs
Saved: results/gap/additive_<run>.json, additive_<run>_rows.npz (per-photo single, linear and alignment).
Usage: python additive.py --run T [--smoke]
"""
import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import common as C
from train_ref import load_mlp_state

torch.set_grad_enabled(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="T")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    base = C.load("base")
    cal, te = C.Cache("cal"), C.Cache("test")
    calib = C.calibrate(base, (x for x, _ in cal.batches(25)))
    rows = np.arange(0, 1500, 15) if a.smoke else np.arange(1500)
    NB = 3 if a.smoke else 30
    z0 = torch.from_numpy(te.z[rows]).double()
    p = F.softmax(z0, 1)

    def metric(e, f=None):
        f = e if f is None else f
        return (p * e * f).sum(1) - (p * e).sum(1) * (p * f).sum(1)

    def kl(z):
        lp = F.log_softmax(z0, 1)
        return (lp.exp() * (lp - F.log_softmax(z, 1))).sum(1)

    res = {}
    out_rows = {}
    for name in ([a.only] if a.only else ["B", "T"]):
        ck = torch.load(os.path.join(C.OUT, f"{a.run}_build.pt" if name == "B" else f"{a.run}.pt"), weights_only=False)
        mX = copy.deepcopy(base)
        C.build(mX, calib, ck["frac"], ck["how"])
        load_mlp_state(mX, ck["state"])
        D = []
        single = []
        for g in range(NB):
            mH = copy.deepcopy(base)
            bh, bx = C.blocks(mH)[g], C.blocks(mX)[g]
            bh.block[3], bh.block[5] = copy.deepcopy(bx.block[3]), copy.deepcopy(bx.block[5])
            zs = []
            for s in range(0, len(rows), 50):
                x, _ = te.get(rows[s:s + 50])
                zs.append(C.tail(mH, x).cpu().double())
            z = torch.cat(zs)
            d = z - z0
            d = d - d.mean(1, keepdim=True)
            D.append(d.float())
            single.append(kl(z))
            del mH
            print(f"{name} block {g}: single KL {float(single[-1].mean()):.4f}", flush=True)
        D = torch.stack(D).double()                              # (NB, n, 1000)
        S = torch.stack(single)                                  # (NB, n)
        tot = D.sum(0)
        lin = kl(z0 + tot)
        num = metric(tot)
        den = sum(metric(D[g]) for g in range(NB))
        align = num / den.clamp_min(1e-12)
        nrm = torch.stack([metric(D[g]).clamp_min(1e-12).sqrt() for g in range(NB)])
        cos = {}
        for lag in (1, 5, 15):
            cs = [float((metric(D[g], D[g + lag]) / (nrm[g] * nrm[g + lag])).mean()) for g in range(NB - lag)]
            cos[lag] = float(np.mean(cs)) if cs else None
        res[name] = dict(single_sum=float(S.sum(0).mean()), linear=float(lin.mean()), align_median=float(align.median()),
                         align_mean_ratio=float(num.mean() / den.mean()), cos_lag=cos,
                         single=S.mean(1).tolist())
        out_rows[f"{name}_single"] = S.sum(0).numpy()
        out_rows[f"{name}_linear"] = lin.numpy()
        out_rows[f"{name}_align"] = align.numpy()
        print(name, {k: v for k, v in res[name].items() if k != "single"}, flush=True)
    tag = a.run + ("_smoke" if a.smoke else "")
    json.dump(res, open(os.path.join(C.OUT, f"additive_{tag}.json"), "w"), indent=1)
    np.savez(os.path.join(C.OUT, f"additive_{tag}_rows.npz"), **out_rows)


if __name__ == "__main__":
    main()
