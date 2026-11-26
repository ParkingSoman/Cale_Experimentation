"""N5 step 6b: the fast switch. At install, each block's hidden neurons are reordered by rank (W1 rows, b1, W2 columns:
the network computes exactly the same function). Then a size is a prefix and each tier a contiguous range, so the
switch uses slices only. Same config as seqtree.py 'ind' (4 tiers, 8 parents, 1,000 calibration photos, one per class).
Measures: KL / top-1 at keep 50% and 25% (must match seqtree ind), warm switch and undo times, undo error,
and the time of one forward pass with the full layer against a truly sliced layer (the compute saving).
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from abstree import parents
from common import DEV, OUT, blocks, kl_rows, load, net, split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pair"))
from distill import D  # noqa: E402

torch.set_grad_enabled(False)
TIERS, S_PAR = 4, 8


def sync():
    if DEV == "mps":
        torch.mps.synchronize()


def main():
    ph = net.Photos("read")
    fit = split(ph.lab)
    cal = np.array(sorted([np.where((ph.lab == c) & fit)[0][7] for c in range(1000)]))
    te = np.load(os.path.join(D, "photos_test.npy"))[:500]
    lab = torch.tensor(ph.lab[te])
    m = load("base")
    BL = blocks(m)
    NB = len(BL)
    acc = [dict(n=0, s=0, ss=0) for _ in range(NB)]
    hs = []
    for k, blk in enumerate(BL):
        def f(mod, i, o, k=k):
            g = o.reshape(-1, o.shape[-1]).float()
            acc[k]["n"] += g.shape[0]
            acc[k]["s"] = acc[k]["s"] + g.sum(0).cpu().double()
            acc[k]["ss"] = acc[k]["ss"] + (g.T @ g).cpu().double()
        hs.append(blk.block[4].register_forward_hook(f))
    for a in range(0, len(cal), 25):
        m(ph.tensor(cal[a:a + 25]).to(DEV))
    for h_ in hs:
        h_.remove()
    ref = [m(ph.tensor(te[a:a + 50]).to(DEV)) for a in range(0, len(te), 50)]
    cfg = []
    for k, blk in enumerate(BL):
        a = acc[k]
        mu = a["s"] / a["n"]
        K = a["ss"] / a["n"] - mu[:, None] * mu[None, :]
        ls = blk.layer_scale.detach().view(-1).cpu().double()
        order = torch.argsort(-(K.diagonal().clamp_min(0).sqrt() * (ls[:, None] * blk.block[5].weight.cpu().double()).norm(dim=0)))
        h = len(order)
        # install: reorder neurons by rank
        blk.block[3].weight.copy_(blk.block[3].weight[order.to(DEV)])
        blk.block[3].bias.copy_(blk.block[3].bias[order.to(DEV)])
        blk.block[5].weight.copy_(blk.block[5].weight[:, order.to(DEV)])
        Kp = K[order][:, order]
        tier = torch.arange(h) * TIERS // h  # in the new order, rank = position
        pid, pc = parents(Kp, tier, smax=S_PAR)[S_PAR]
        cfg.append(dict(pid=pid.to(DEV), c=pc.float().to(DEV), mu=mu[order].float().to(DEV), h=h))
        acc[k] = None
    # check: reordering changes nothing
    d0 = max(float((m(ph.tensor(te[a:a + 50]).to(DEV)) - ref[i]).abs().max()) for i, a in enumerate(range(0, 100, 50)))
    print(f"reorder check: max logit change {d0:.2e}", flush=True)
    orig = [(b.block[5].weight.detach().clone(), b.block[5].bias.detach().clone()) for b in BL]
    keep = [None] * NB
    for k, blk in enumerate(BL):  # evaluation mask = 'use the first kk neurons'
        blk.block[4].register_forward_hook(lambda mod, i, o, k=k: o * keep[k] if keep[k] is not None else o)

    def frac(fr, k):
        return fr[k] if isinstance(fr, (list, tuple)) else fr

    def switch(fr):
        for k, blk in enumerate(BL):
            c = cfg[k]
            W, b = blk.block[5].weight, blk.block[5].bias
            h = c["h"]
            kk = int(round(frac(fr, k) * h))
            base_mean = W @ c["mu"]
            for t in range(TIERS - 1, -1, -1):
                lo, hi = max(t * h // TIERS, kk), (t + 1) * h // TIERS
                if lo >= hi:
                    continue
                contrib = (W[:, lo:hi, None] * c["c"][lo:hi][None]).reshape(W.shape[0], -1)
                W.index_add_(1, c["pid"][lo:hi].reshape(-1), contrib)
            b += base_mean - W[:, :kk] @ c["mu"][:kk]

    def undo(fr):
        for k in range(NB - 1, -1, -1):
            c = cfg[k]
            W, b = BL[k].block[5].weight, BL[k].block[5].bias
            h = c["h"]
            kk = int(round(frac(fr, k) * h))
            for t in range(TIERS):
                lo, hi = max(t * h // TIERS, kk), (t + 1) * h // TIERS
                if lo >= hi:
                    continue
                contrib = (W[:, lo:hi, None] * c["c"][lo:hi][None]).reshape(W.shape[0], -1)
                W.index_add_(1, c["pid"][lo:hi].reshape(-1), -contrib)
            b.copy_(orig[k][1])  # one stored bias per size in practice (d numbers per block)

    def evaluate(fr):
        for k in range(NB):
            kk = int(round(frac(fr, k) * cfg[k]["h"]))
            mk = torch.zeros(cfg[k]["h"], device=DEV)
            mk[:kk] = 1
            keep[k] = mk
        kl, cc = [], 0
        for i, a in enumerate(range(0, len(te), 50)):
            y = m(ph.tensor(te[a:a + 50]).to(DEV))
            kl.append(kl_rows(ref[i], y).cpu())
            cc += int((y.argmax(1).cpu() == lab[a:a + 50]).sum())
        for k in range(NB):
            keep[k] = None
        return float(torch.cat(kl).mean()), cc / len(te)

    res = {"reorder_max_logit_change": d0}
    # uneven budgets from step 0's single-block costs (keep 50%, value): f_b = clip(t * (cost_b / mean)^g), same total
    cost = np.array(json.load(open(os.path.join(OUT, "diag.json")))["single_value"])
    hs_ = np.array([cfg[k]["h"] for k in range(NB)], float)

    def budget(Ftot, g):
        w = (cost / cost.mean()) ** g
        lo, hi = 0.0, 10.0
        for _ in range(60):
            t = (lo + hi) / 2
            f = np.clip(t * w, 0.1, 0.95)
            if (f * hs_).sum() > Ftot * hs_.sum():
                hi = t
            else:
                lo = t
        return np.clip(lo * w, 0.1, 0.95).tolist()
    plans = [(0.5, 0.5), (0.25, 0.25)]
    for Ftot in (0.5, 0.25):
        for g in (-0.5, -1.0):
            plans.append((f"{Ftot}_cost^{g}", budget(Ftot, g)))
    for name, fr in plans:
        if isinstance(fr, list):
            kl, top1 = (switch(fr), evaluate(fr))[1]
            undo(fr)
            res[f"budget_{name}"] = dict(kl=kl, top1=top1, fractions=[round(x, 3) for x in fr])
            print(f"budget {name}: KL {kl:.4f} top-1 {top1:.3f}; fractions {[round(x, 2) for x in fr]}", flush=True)
    for fr in (0.5, 0.25):
        ts, us = [], []
        for rep in range(5):
            sync()
            t0 = time.time()
            switch(fr)
            sync()
            ts.append(1000 * (time.time() - t0))
            if rep == 0:
                kl, top1 = evaluate(fr)
            t0 = time.time()
            undo(fr)
            sync()
            us.append(1000 * (time.time() - t0))
        err = max(float((BL[k].block[5].weight - orig[k][0]).abs().max() / orig[k][0].abs().max()) for k in range(NB))
        res[f"keep{fr}"] = dict(kl=kl, top1=top1, switch_ms=ts, undo_ms=us, undo_max_rel_err=err)
        print(f"keep {fr}: KL {kl:.4f} top-1 {top1:.3f}; switch ms {[round(x, 1) for x in ts]}; "
              f"undo ms {[round(x, 1) for x in us]}; undo error {err:.1e}", flush=True)
    # compute saving: one forward pass of all 30 MLPs, full width against a true prefix slice (keep 50%)
    x = ph.tensor(te[:50]).to(DEV)
    U = {}
    hk = [blk.block[2].register_forward_hook(lambda mod, i, o, k=k: U.__setitem__(k, o)) for k, blk in enumerate(BL)]
    m(x)
    for h_ in hk:
        h_.remove()

    def mlp_time(fr, reps=10):
        sync()
        t0 = time.time()
        for _ in range(reps):
            for k, blk in enumerate(BL):
                kk = int(round(fr * cfg[k]["h"]))
                g = F.gelu(F.linear(U[k], blk.block[3].weight[:kk], blk.block[3].bias[:kk]))
                F.linear(g, blk.block[5].weight[:, :kk], blk.block[5].bias)
        sync()
        return 1000 * (time.time() - t0) / reps
    mlp_time(1.0, 2)
    res["mlp_ms_full"], res["mlp_ms_half"], res["mlp_ms_quarter"] = mlp_time(1.0), mlp_time(0.5), mlp_time(0.25)
    print(f"30 MLPs on 50 photos: full {res['mlp_ms_full']:.1f} ms, keep 50% {res['mlp_ms_half']:.1f} ms, "
          f"keep 25% {res['mlp_ms_quarter']:.1f} ms", flush=True)
    json.dump(res, open(os.path.join(OUT, "fastswitch.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
