"""Debug for step 6: is the tiered result (worse than mean fill) a bug or the tier constraint?
Same O statistics for every row. Keep 50%. Rows:
  mean fill | tiers=T, triangular switch (abstree) | tiers=T, index_add switch (seqtree) for T in (16, 128, h)
"""
import os
import sys

import numpy as np
import torch

from abstree import parents
from common import DEV, blocks, kl_rows, load, net, split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pair"))
from distill import D  # noqa: E402

torch.set_grad_enabled(False)
FR, S = 0.5, 8


def main():
    ph = net.Photos("read")
    fit = split(ph.lab)
    cal = np.array(sorted([np.where((ph.lab == c) & fit)[0][7] for c in range(1000)]))
    cal = np.sort(np.random.default_rng(0).choice(cal, 300, replace=False))
    te = np.load(os.path.join(D, "photos_test.npy"))[:200]
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
    mask = [None] * NB
    for k, blk in enumerate(BL):
        blk.block[4].register_forward_hook(lambda mod, i, o, k=k: o * mask[k] if mask[k] is not None else o)
    orig = [(b.block[5].weight.detach().clone(), b.block[5].bias.detach().clone()) for b in BL]
    ref = [m(ph.tensor(te[a:a + 50]).to(DEV)) for a in range(0, len(te), 50)]

    def ev():
        return float(torch.cat([kl_rows(ref[i], m(ph.tensor(te[a:a + 50]).to(DEV))).cpu()
                                for i, a in enumerate(range(0, len(te), 50))]).mean())

    def restore():
        for k, b in enumerate(BL):
            b.block[5].weight.copy_(orig[k][0])
            b.block[5].bias.copy_(orig[k][1])
            mask[k] = None

    info = []
    for k in range(NB):
        a = acc[k]
        mu = a["s"] / a["n"]
        K = a["ss"] / a["n"] - mu[:, None] * mu[None, :]
        Wb = BL[k].layer_scale.detach().view(-1).cpu().double()[:, None] * orig[k][0].cpu().double()
        order = torch.argsort(-(K.diagonal().clamp_min(0).sqrt() * Wb.norm(dim=0)))
        h = len(order)
        rp = torch.empty(h, dtype=torch.long)
        rp[order] = torch.arange(h)
        info.append(dict(K=K, mu=mu.float(), order=order, rp=rp))

    def tri_switch(k, pid, pc):
        W, b = BL[k].block[5].weight, BL[k].block[5].bias
        order = info[k]["order"].to(DEV)
        h = W.shape[1]
        kk = int(round(FR * h))
        Sx, Dd = order[:kk], order[kk:]
        mu = info[k]["mu"].to(DEV)
        pos = torch.full((h,), -1, dtype=torch.long, device=DEV)
        pos[Dd] = torch.arange(len(Dd), device=DEV)
        posS = torch.full((h,), -1, dtype=torch.long, device=DEV)
        posS[Sx] = torch.arange(kk, device=DEV)
        pd, cd = pid.to(DEV)[Dd], pc.float().to(DEV)[Dd]
        rows = torch.arange(len(Dd), device=DEV)[:, None].expand_as(pd)
        CDD = torch.zeros(len(Dd), len(Dd), device=DEV)
        CDS = torch.zeros(len(Dd), kk, device=DEV)
        inD, inS = pos[pd] >= 0, posS[pd] >= 0
        CDD.index_put_((rows[inD], pos[pd][inD]), cd[inD], accumulate=True)
        CDS.index_put_((rows[inS], posS[pd][inS]), cd[inS], accumulate=True)
        upper_mass = float(torch.triu(CDD).abs().sum())
        X = torch.linalg.solve_triangular(torch.eye(len(Dd), device=DEV) - CDD, CDS, upper=False)
        WD = W[:, Dd]
        dW = WD @ X
        W[:, Sx] += dW
        b += WD @ mu[Dd] - dW @ mu[Sx]
        W[:, Dd] = 0
        return upper_mass

    def add_switch(k, pid, pc, tier, T):
        W, b = BL[k].block[5].weight, BL[k].block[5].bias
        order = info[k]["order"]
        h = W.shape[1]
        kk = int(round(FR * h))
        Sx, Dd = order[:kk], order[kk:]
        mu = info[k]["mu"].to(DEV)
        base_mean = W @ mu
        pid_d, pc_d = pid.to(DEV), pc.float().to(DEV)
        for t in range(T - 1, -1, -1):
            d = Dd[tier[Dd] == t]
            if len(d) == 0:
                continue
            d = d.to(DEV)
            contrib = (W[:, d][:, :, None] * pc_d[d][None]).reshape(W.shape[0], -1)
            W.index_add_(1, pid_d[d].reshape(-1), contrib)
        Sd = Sx.to(DEV)
        b += base_mean - W[:, Sd] @ mu[Sd]
        mk = torch.zeros(h, device=DEV)
        mk[Sd] = 1
        mask[k] = mk

    restore()
    for k in range(NB):
        h = BL[k].block[5].weight.shape[1]
        add_switch(k, torch.zeros(h, S, dtype=torch.long), torch.zeros(h, S), torch.zeros(h, dtype=torch.long), 1)
    print(f"mean fill (add path): {ev():.4f}", flush=True)
    for T in (16, 128, "rank"):
        par = []
        for k in range(NB):
            h = len(info[k]["order"])
            tier = info[k]["rp"] if T == "rank" else info[k]["rp"] * T // h
            pid, pc = parents(info[k]["K"], tier, smax=S)[S]
            par.append((pid, pc, tier))
        restore()
        um = sum(tri_switch(k, p[0], p[1]) for k, p in enumerate(par))
        r1 = ev()
        restore()
        if T != "rank":
            for k, p in enumerate(par):
                add_switch(k, p[0], p[1], p[2], T)
            r2 = ev()
        else:
            r2 = float("nan")
        print(f"tiers {T}: triangular {r1:.4f} (upper-triangle mass {um:.3g}), index_add {r2:.4f}", flush=True)
        restore()


if __name__ == "__main__":
    main()
