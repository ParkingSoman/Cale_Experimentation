"""N5 step 6: tiered absorption tree, fitted independently (on the original network O) or SEQUENTIALLY (block by block,
on the shrunk network C, to rebuild O's output). All sizes are in-place computations on the base weights + a config.

Per block, h hidden neurons, ranked by value on O; tier = rank * TIERS // h. Parents of a neuron: up to s neurons of
strictly higher tiers (OMP on O's covariance; abstree.parents with a tier mask). A config for keep fraction f holds:
  mu_O (h), mu_C (h), a (scale of each kept neuron), c (h, s) coefficients.   Model:  g_x^O ~ mu_x^O + sum_p c_xp (g_p - mu_p^C)
  ind: mu_C = mu_O, a = 1, c from O's covariance (as in step 5)
  seq: for block b, statistics of g^O (target) and g^C (shrunk net so far, block b unchanged); c_d = regression of g_d^O on
       its parents' g^C; a_s = cov(g_s^O, g_s^C) / var(g_s^C)
Switch (in place): W_S *= a; for tiers from lowest to highest, every dropped neuron d adds c_d x W_d to its parents'
columns (one index_add per tier); b <- b + W_orig mu_O - W_S' mu_C[S]. Dropped neurons are masked at run time, their columns
keep their pushed values, so undo = the same steps in reverse (subtract, then divide by a).
Usage: python seqtree.py [n_cal] [n_test]
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


class Stop(Exception):
    pass


def main():
    ncal = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    nte = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    ph = net.Photos("read")
    fit = split(ph.lab)
    cal_all = np.array(sorted([np.where((ph.lab == c) & fit)[0][7] for c in range(1000)]))
    cal = np.sort(np.random.default_rng(0).choice(cal_all, ncal, replace=False)) if ncal < 1000 else cal_all
    te = np.load(os.path.join(D, "photos_test.npy"))[:nte]
    lab = torch.tensor(ph.lab[te])
    O, C = load("base"), load("base")
    BO, BC = blocks(O), blocks(C)
    NB = len(BC)
    orig = [(b.block[5].weight.detach().clone(), b.block[5].bias.detach().clone()) for b in BC]
    mask = [None] * NB  # run-time mask of dropped neurons (multiplies the GELU output)
    for k, blk in enumerate(BC):
        blk.block[4].register_forward_hook(lambda mod, i, o, k=k: o * mask[k] if mask[k] is not None else o)

    def stats(model, BLs, want_o=False):
        """Per block: mean and covariance of GELU outputs on the calibration photos (float64 cpu)."""
        acc = [dict(n=0, s=0, ss=0) for _ in range(NB)]
        hs = []
        for k, blk in enumerate(BLs):
            def f(mod, i, o, k=k):
                g = o.reshape(-1, o.shape[-1]).float()
                a = acc[k]
                a["n"] += g.shape[0]
                a["s"] = a["s"] + g.sum(0).cpu().double()
                a["ss"] = a["ss"] + (g.T @ g).cpu().double()
            hs.append(blk.block[4].register_forward_hook(f))
        for a in range(0, len(cal), 25):
            model(ph.tensor(cal[a:a + 25]).to(DEV))
        for h in hs:
            h.remove()
        return acc

    t0 = time.time()
    accO = stats(O, BO)
    cfg = []
    for k in range(NB):
        a = accO[k]
        mu = a["s"] / a["n"]
        K = a["ss"] / a["n"] - mu[:, None] * mu[None, :]
        Wb = orig[k][0].cpu().double()
        h = Wb.shape[1]
        order = torch.argsort(-(K.diagonal().clamp_min(0).sqrt() * (BC[k].layer_scale.detach().view(-1).cpu().double()[:, None] * Wb).norm(dim=0)))
        rank_pos = torch.empty(h, dtype=torch.long)
        rank_pos[order] = torch.arange(h)
        tier = rank_pos * TIERS // h
        par = parents(K, tier * h + rank_pos * 0, smax=S_PAR)  # higher = strictly lower tier number
        pid, pc = par[S_PAR]
        cfg.append(dict(order=order, tier=tier, pid=pid, c_ind=pc.float(), mu=mu.float(), K=None))
        accO[k] = None
    del accO
    print(f"O stats + tiered parents {time.time() - t0:.0f}s", flush=True)

    ref = [O(ph.tensor(te[a:a + 50]).to(DEV)) for a in range(0, len(te), 50)]

    def evaluate():
        kl, c = [], 0
        for i, a in enumerate(range(0, len(te), 50)):
            y = C(ph.tensor(te[a:a + 50]).to(DEV))
            kl.append(kl_rows(ref[i], y).cpu())
            c += int((y.argmax(1).cpu() == lab[a:a + 50]).sum())
        return torch.cat(kl).numpy(), c / len(te)

    def restore():
        for k, blk in enumerate(BC):
            blk.block[5].weight.copy_(orig[k][0])
            blk.block[5].bias.copy_(orig[k][1])
            mask[k] = None

    def plan(k, fr):
        """Device tensors for block k at keep fraction fr: kept ids, dropped ids grouped by tier (lowest tier first)."""
        c = cfg[k]
        h = len(c["order"])
        kk = int(round(fr * h))
        S, Dd = c["order"][:kk], c["order"][kk:]
        groups = []
        for t in range(TIERS - 1, -1, -1):
            d = Dd[c["tier"][Dd] == t]
            if len(d):
                groups.append(d.to(DEV))
        return S.to(DEV), Dd.to(DEV), groups

    def switch_block(k, fr, conf, P):
        W, b = BC[k].block[5].weight, BC[k].block[5].bias
        S, Dd, groups = P
        base_mean = W @ conf["muO"]  # W_orig mu_O
        W[:, S] *= conf["a"][None, :]
        for d in groups:
            pid, c = conf["pid"][d], conf["c"][d]  # (n, s)
            contrib = (W[:, d][:, :, None] * c[None]).reshape(W.shape[0], -1)
            W.index_add_(1, pid.reshape(-1), contrib)
        b += base_mean - W[:, S] @ conf["muC"][S]
        m = torch.zeros(W.shape[1], device=DEV)
        m[S] = 1
        mask[k] = m

    def unswitch_block(k, fr, conf, P, b_before):
        W, b = BC[k].block[5].weight, BC[k].block[5].bias
        S, Dd, groups = P
        for d in reversed(groups):
            pid, c = conf["pid"][d], conf["c"][d]
            contrib = (W[:, d][:, :, None] * c[None]).reshape(W.shape[0], -1)
            W.index_add_(1, pid.reshape(-1), -contrib)
        W[:, S] /= conf["a"][None, :]
        b.copy_(b_before)  # the bias delta is one stored vector per size, put back
        mask[k] = None

    def dev_conf(k, c_coef, a=None, muC=None):
        h = len(cfg[k]["order"])
        return dict(pid=cfg[k]["pid"].to(DEV), c=c_coef.to(DEV), muO=cfg[k]["mu"].to(DEV),
                    muC=(cfg[k]["mu"] if muC is None else muC).to(DEV),
                    a=(torch.ones(h) if a is None else a).float().to(DEV))

    res = {}
    for fr in (0.5, 0.25):
        P = [plan(k, fr) for k in range(NB)]
        # independent tiered config
        restore()
        confs = [dev_conf(k, cfg[k]["c_ind"]) for k in range(NB)]
        confs = [dict(cf, a=cf["a"][P[k][0]]) for k, cf in enumerate(confs)]
        if DEV == "mps":
            torch.mps.synchronize()
        t1 = time.time()
        for k in range(NB):
            switch_block(k, fr, confs[k], P[k])
        if DEV == "mps":
            torch.mps.synchronize()
        dt = 1000 * (time.time() - t1)
        kl, top1 = evaluate()
        res[f"ind_{fr}"] = dict(kl=float(kl.mean()), top1=top1, switch_ms=dt)
        print(f"keep {fr} tiered, independent: KL {kl.mean():.4f} top-1 {top1:.3f} switch {dt:.1f} ms", flush=True)
        # undo check
        bb = [orig[k][1] for k in range(NB)]
        t1 = time.time()
        for k in range(NB - 1, -1, -1):
            unswitch_block(k, fr, confs[k], P[k], bb[k])
        if DEV == "mps":
            torch.mps.synchronize()
        err = max(float((BC[k].block[5].weight - orig[k][0]).abs().max() / orig[k][0].abs().max()) for k in range(NB))
        res[f"ind_{fr}"]["undo_ms"] = 1000 * (time.time() - t1)
        res[f"ind_{fr}"]["undo_max_rel_err"] = err
        print(f"  undo {res[f'ind_{fr}']['undo_ms']:.1f} ms, max relative weight error {err:.2e}", flush=True)
        warm = []
        for rep in range(3):  # warm timings: switch, then undo
            torch.mps.synchronize() if DEV == "mps" else None
            t1 = time.time()
            for k in range(NB):
                switch_block(k, fr, confs[k], P[k])
            torch.mps.synchronize() if DEV == "mps" else None
            warm.append(1000 * (time.time() - t1))
            for k in range(NB - 1, -1, -1):
                unswitch_block(k, fr, confs[k], P[k], bb[k])
        res[f"ind_{fr}"]["switch_ms_warm"] = warm
        print(f"  warm switch times (ms): {[round(w, 1) for w in warm]}", flush=True)
        # sequential config: block by block on C
        restore()
        t2 = time.time()
        seq_confs = []
        for k in range(NB):
            H = {}
            hO = BO[k].block[4].register_forward_hook(lambda mod, i, o: (H.__setitem__("o", o), (_ for _ in ()).throw(Stop())))
            hC = BC[k].block[4].register_forward_hook(lambda mod, i, o: (H.__setitem__("c", o), (_ for _ in ()).throw(Stop())))
            n, sO, sC, CC, OC = 0, 0, 0, 0, 0
            for a in range(0, len(cal), 25):
                x = ph.tensor(cal[a:a + 25]).to(DEV)
                for model in (O, C):
                    try:
                        model(x)
                    except Stop:
                        pass
                gO = H["o"].reshape(-1, H["o"].shape[-1]).float()
                gC = H["c"].reshape(-1, H["c"].shape[-1]).float()
                n += gO.shape[0]
                sO = sO + gO.sum(0).cpu().double()
                sC = sC + gC.sum(0).cpu().double()
                CC = CC + (gC.T @ gC).cpu().double()
                OC = OC + (gO.T @ gC).cpu().double()
            hO.remove()
            hC.remove()
            muO, muC = sO / n, sC / n
            KCC = CC / n - muC[:, None] * muC[None, :]
            KOC = OC / n - muO[:, None] * muC[None, :]
            pid = cfg[k]["pid"]  # (h, s)
            h = pid.shape[0]
            Kpp = KCC[pid[:, :, None], pid[:, None, :]]  # (h, s, s)
            Kdp = KOC[torch.arange(h)[:, None], pid]  # (h, s)
            reg = 1e-4 * KCC.diagonal().mean()
            same = (pid[:, :, None] == pid[:, None, :]).double()  # duplicate picks of an invalid pool
            cc = torch.linalg.solve(Kpp + reg * torch.eye(S_PAR, dtype=Kpp.dtype) + 1e3 * reg * (same - torch.eye(S_PAR, dtype=Kpp.dtype)).clamp_min(0), Kdp)
            cc = torch.where(cfg[k]["c_ind"] != 0, cc, torch.zeros_like(cc))
            S = P[k][0].cpu()
            a_s = KOC.diagonal()[S] / KCC.diagonal()[S].clamp_min(1e-12)
            conf = dev_conf(k, cc.float(), a=a_s, muC=muC.float())
            conf["muO"] = muO.float().to(DEV)
            seq_confs.append(conf)
            switch_block(k, fr, conf, P[k])
            del KCC, KOC, CC, OC
            print(f"  seq keep {fr} block {k} [{time.time() - t2:.0f}s]", flush=True)
        kl, top1 = evaluate()
        res[f"seq_{fr}"] = dict(kl=float(kl.mean()), top1=top1, fit_s=time.time() - t2)
        print(f"keep {fr} tiered, sequential: KL {kl.mean():.4f} top-1 {top1:.3f}", flush=True)
        json.dump(res, open(os.path.join(OUT, f"seqtree_{ncal}.json"), "w"), indent=1)
    restore()


if __name__ == "__main__":
    main()
