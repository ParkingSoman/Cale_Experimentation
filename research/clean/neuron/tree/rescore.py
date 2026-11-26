"""N5 rescore: all 1,500 test photos (938 classes; the first 500 cover only classes 0-329). 1,000 calibration photos.
Rows at keep 50% and 25% (and 40%, off a tier boundary): mean fill, tree (4 tiers, 8 parents, O covariance), tree with
RANDOM parents (8 random higher-tier neurons, coefficients fitted the same way), dense per-size re-solve (ceiling).
Paired bootstrap intervals on per-photo KL. 100 switch + undo cycles: weight drift. Switch time split by step.
"""
import json
import os
import sys
import time

import numpy as np
import torch

from abstree import parents
from common import DEV, OUT, blocks, kl_rows, load, net, split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pair"))
from distill import D  # noqa: E402

torch.set_grad_enabled(False)
TIERS, S_PAR = 4, 8


def sync():
    if DEV == "mps":
        torch.mps.synchronize()


def ci(d):
    bs = [d[np.random.default_rng(i).integers(0, len(d), len(d))].mean() for i in range(2000)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    ph = net.Photos("read")
    fit = split(ph.lab)
    cal = np.array(sorted([np.where((ph.lab == c) & fit)[0][7] for c in range(1000)]))
    te = np.load(os.path.join(D, "photos_test.npy"))
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
    base_top1 = float(np.mean(np.concatenate([(r.argmax(1).cpu() == lab[i * 50:(i + 1) * 50]).numpy() for i, r in enumerate(ref)])))
    print(f"Base top-1 on all 1,500: {base_top1:.3f}", flush=True)
    rng = np.random.default_rng(0)
    cfg = []
    for k, blk in enumerate(BL):
        a = acc[k]
        mu = a["s"] / a["n"]
        K = a["ss"] / a["n"] - mu[:, None] * mu[None, :]
        ls = blk.layer_scale.detach().view(-1).cpu().double()
        Wb = ls[:, None] * blk.block[5].weight.detach().cpu().double()
        order = torch.argsort(-(K.diagonal().clamp_min(0).sqrt() * Wb.norm(dim=0)))
        h = len(order)
        od = order.to(DEV)
        blk.block[3].weight.copy_(blk.block[3].weight[od])
        blk.block[3].bias.copy_(blk.block[3].bias[od])
        blk.block[5].weight.copy_(blk.block[5].weight[:, od])
        Kp = K[order][:, order]
        tier = torch.arange(h) * TIERS // h
        pid, pc = parents(Kp, tier, smax=S_PAR)[S_PAR]
        # random parents: S_PAR random neurons of strictly higher tiers, least squares on O's covariance
        rid = torch.zeros(h, S_PAR, dtype=torch.long)
        rc = torch.zeros(h, S_PAR, dtype=torch.float64)
        for i in range(h // TIERS, h):
            lim = int(tier[i]) * h // TIERS
            q = torch.from_numpy(rng.choice(lim, S_PAR, replace=False))
            rid[i] = q
            A = Kp[q][:, q] + 1e-8 * Kp.diagonal().mean() * torch.eye(S_PAR, dtype=torch.float64)
            rc[i] = torch.linalg.solve(A, Kp[q, i])
        Wbp = Wb[:, order]
        dense = {}
        for fr in (0.5, 0.4, 0.25):
            kk = int(round(fr * h))
            KSS = Kp[:kk, :kk] + 1e-6 * Kp.diagonal().mean() * torch.eye(kk, dtype=torch.float64)
            Wn = torch.zeros_like(Wbp)
            Wn[:, :kk] = torch.linalg.solve(KSS, Kp[:kk] @ Wbp.T).T
            bshift = Wbp @ mu[order] - Wn @ mu[order]
            dense[fr] = ((Wn / ls[:, None]).float(), (bshift / ls).float())
        cfg.append(dict(pid=pid.to(DEV), c=pc.float().to(DEV), rid=rid.to(DEV), rc=rc.float().to(DEV),
                        mu=mu[order].float().to(DEV), h=h, dense=dense))
        acc[k] = None
        print(f"  block {k} config done", flush=True)
    orig = [(b.block[5].weight.detach().clone(), b.block[5].bias.detach().clone()) for b in BL]
    keep = [None] * NB
    for k, blk in enumerate(BL):
        blk.block[4].register_forward_hook(lambda mod, i, o, k=k: o * keep[k] if keep[k] is not None else o)
    T = dict(mean=0.0, tiers=0.0, bias=0.0)

    def switch(fr, which, timing=False):
        for k, blk in enumerate(BL):
            c = cfg[k]
            W, b = blk.block[5].weight, blk.block[5].bias
            h = c["h"]
            kk = int(round(fr * h))
            t0 = time.time()
            base_mean = W @ c["mu"]
            if timing:
                sync()
                T["mean"] += time.time() - t0
                t0 = time.time()
            if which != "meanfill":
                pid, cc = (c["pid"], c["c"]) if which == "tree" else (c["rid"], c["rc"])
                for t in range(TIERS - 1, -1, -1):
                    lo, hi = max(t * h // TIERS, kk), (t + 1) * h // TIERS
                    if lo < hi:
                        W.index_add_(1, pid[lo:hi].reshape(-1), (W[:, lo:hi, None] * cc[lo:hi][None]).reshape(W.shape[0], -1))
            if timing:
                sync()
                T["tiers"] += time.time() - t0
                t0 = time.time()
            b += base_mean - W[:, :kk] @ c["mu"][:kk]
            if timing:
                sync()
                T["bias"] += time.time() - t0

    def undo(fr, which):
        for k in range(NB - 1, -1, -1):
            c = cfg[k]
            W, b = BL[k].block[5].weight, BL[k].block[5].bias
            h = c["h"]
            kk = int(round(fr * h))
            if which != "meanfill":
                pid, cc = (c["pid"], c["c"]) if which == "tree" else (c["rid"], c["rc"])
                for t in range(TIERS):
                    lo, hi = max(t * h // TIERS, kk), (t + 1) * h // TIERS
                    if lo < hi:
                        W.index_add_(1, pid[lo:hi].reshape(-1), -(W[:, lo:hi, None] * cc[lo:hi][None]).reshape(W.shape[0], -1))
            b.copy_(orig[k][1])  # the base bias, kept as is (d numbers per block)

    def evaluate(fr):
        for k in range(NB):
            mk = torch.zeros(cfg[k]["h"], device=DEV)
            mk[:int(round(fr * cfg[k]["h"]))] = 1
            keep[k] = mk
        kl, cc = [], []
        for i, a in enumerate(range(0, len(te), 50)):
            y = m(ph.tensor(te[a:a + 50]).to(DEV))
            kl.append(kl_rows(ref[i], y).cpu())
            cc.append((y.argmax(1).cpu() == lab[a:a + 50]).float())
        for k in range(NB):
            keep[k] = None
        return torch.cat(kl).numpy(), torch.cat(cc).numpy()

    res = dict(base_top1=base_top1, n_test=int(len(te)), n_classes_test=int(len(set(ph.lab[te].tolist()))))
    rows = {}
    for fr in (0.5, 0.4, 0.25):
        for which in ("meanfill", "tree", "randpar"):
            switch(fr, which)
            kl, ok = evaluate(fr)
            undo(fr, which)
            rows[f"{which}_{fr}"] = (kl, ok)
            res[f"{which}_{fr}"] = dict(kl=float(kl.mean()), top1=float(ok.mean()))
            print(f"keep {fr} {which}: KL {kl.mean():.4f} top-1 {ok.mean():.3f}", flush=True)
        for k, blk in enumerate(BL):
            blk.block[5].weight.copy_(cfg[k]["dense"][fr][0].to(DEV))
            blk.block[5].bias.copy_(orig[k][1] + cfg[k]["dense"][fr][1].to(DEV))
        kl, ok = evaluate(fr)
        for k, blk in enumerate(BL):
            blk.block[5].weight.copy_(orig[k][0])
            blk.block[5].bias.copy_(orig[k][1])
        rows[f"dense_{fr}"] = (kl, ok)
        res[f"dense_{fr}"] = dict(kl=float(kl.mean()), top1=float(ok.mean()))
        print(f"keep {fr} dense: KL {kl.mean():.4f} top-1 {ok.mean():.3f}", flush=True)
        for a, b in (("tree", "meanfill"), ("tree", "randpar"), ("dense", "tree")):
            dk = rows[f"{a}_{fr}"][0] - rows[f"{b}_{fr}"][0]
            dt = rows[f"{a}_{fr}"][1] - rows[f"{b}_{fr}"][1]
            res[f"{a}-{b}_{fr}"] = dict(kl=float(dk.mean()), kl_ci=ci(dk), top1=float(dt.mean()), top1_ci=ci(dt))
            print(f"  {a} - {b}: KL {dk.mean():+.4f} {ci(dk)}  top-1 {dt.mean():+.4f} {ci(dt)}", flush=True)
        json.dump(res, open(os.path.join(OUT, "rescore.json"), "w"), indent=1)
    # drift over 100 switch + undo cycles, and the timing split
    for k in range(NB):
        T.update(mean=0.0, tiers=0.0, bias=0.0)
    for cyc in range(100):
        switch(0.5, "tree", timing=(cyc >= 90))
        undo(0.5, "tree")
    sync()
    err = max(float((BL[k].block[5].weight - orig[k][0]).abs().max() / orig[k][0].abs().max()) for k in range(NB))
    res["drift_after_100_cycles_max_rel"] = err
    res["switch_split_ms_per_switch"] = {k: 1000 * v / 10 for k, v in T.items()}
    print(f"after 100 cycles: max relative weight drift {err:.2e}; switch split (ms, with a sync after each step): "
          f"{res['switch_split_ms_per_switch']}", flush=True)
    json.dump(res, open(os.path.join(OUT, "rescore.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
