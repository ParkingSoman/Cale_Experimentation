"""Debug for step 6: why does the sequential fit (KL .68) do worse than the independent fit (KL .33) at keep 50%?
First J blocks only, cumulative KL after each block (200 test photos), 1,000 calibration photos (one per class).
Variants of the per-block config (same tiered parents):
  ind      c from O, a = 1, mu_C = mu_O
  seq      c = regression of g^O on parents' g^C, a = cov(g^O, g^C) / var(g^C), mu_C from C
  seq_a1   as seq, a = 1
  seq_mu   c from O, a = 1, only mu_C from C
Also prints quantiles of a.
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
TIERS, S_PAR, FR, J = 4, 8, 0.5, 30
CHECK = (15, 22, 29)


class Stop(Exception):
    pass


def main():
    ph = net.Photos("read")
    fit = split(ph.lab)
    cal = np.array(sorted([np.where((ph.lab == c) & fit)[0][7] for c in range(1000)]))
    te = np.load(os.path.join(D, "photos_test.npy"))[:200]
    O, C = load("base"), load("base")
    BO, BC = blocks(O), blocks(C)
    orig = [(b.block[5].weight.detach().clone(), b.block[5].bias.detach().clone()) for b in BC]
    mask = [None] * J
    for k in range(J):
        BC[k].block[4].register_forward_hook(lambda mod, i, o, k=k: o * mask[k] if mask[k] is not None else o)
    acc = [dict(n=0, s=0, ss=0) for _ in range(J)]
    hs = []
    for k in range(J):
        def f(mod, i, o, k=k):
            g = o.reshape(-1, o.shape[-1]).float()
            acc[k]["n"] += g.shape[0]
            acc[k]["s"] = acc[k]["s"] + g.sum(0).cpu().double()
            acc[k]["ss"] = acc[k]["ss"] + (g.T @ g).cpu().double()
        hs.append(BO[k].block[4].register_forward_hook(f))
    hs.append(BO[J - 1].block[5].register_forward_hook(lambda mod, i, o: (_ for _ in ()).throw(Stop())))
    for a in range(0, len(cal), 25):
        try:
            O(ph.tensor(cal[a:a + 25]).to(DEV))
        except Stop:
            pass
    for h_ in hs:
        h_.remove()
    cfg = []
    for k in range(J):
        a = acc[k]
        mu = a["s"] / a["n"]
        K = a["ss"] / a["n"] - mu[:, None] * mu[None, :]
        ls = BC[k].layer_scale.detach().view(-1).cpu().double()
        order = torch.argsort(-(K.diagonal().clamp_min(0).sqrt() * (ls[:, None] * orig[k][0].cpu().double()).norm(dim=0)))
        h = len(order)
        rp = torch.empty(h, dtype=torch.long)
        rp[order] = torch.arange(h)
        pid, pc = parents(K, rp * TIERS // h, smax=S_PAR)[S_PAR]
        kk = int(round(FR * h))
        cfg.append(dict(order=order, tier=rp * TIERS // h, pid=pid, c=pc.float(), mu=mu.float(), S=order[:kk], Dd=order[kk:],
                        sdO=K.diagonal().clamp_min(0).sqrt()))
    ref = [O(ph.tensor(te[a:a + 50]).to(DEV)) for a in range(0, len(te), 50)]

    def ev():
        return float(torch.cat([kl_rows(ref[i], C(ph.tensor(te[a:a + 50]).to(DEV))).cpu()
                                for i, a in enumerate(range(0, len(te), 50))]).mean())

    def switch(k, c, a_s, muO, muC):
        W, b = BC[k].block[5].weight, BC[k].block[5].bias
        cf = cfg[k]
        S, Dd = cf["S"].to(DEV), cf["Dd"]
        base = W @ muO.to(DEV)
        W[:, S] *= a_s.float().to(DEV)[None]
        pid, cc = cf["pid"].to(DEV), c.float().to(DEV)
        for t in range(TIERS - 1, -1, -1):
            d = Dd[cf["tier"][Dd] == t].to(DEV)
            if len(d):
                W.index_add_(1, pid[d].reshape(-1), (W[:, d][:, :, None] * cc[d][None]).reshape(W.shape[0], -1))
        b += base - W[:, S] @ muC.to(DEV)[S]
        mk = torch.zeros(W.shape[1], device=DEV)
        mk[S] = 1
        mask[k] = mk

    def seq_stats(k):
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
        return muO, muC, CC / n - muC[:, None] * muC[None, :], OC / n - muO[:, None] * muC[None, :]

    for var in ("seq_mu", "seq_match"):
        for k, b in enumerate(BC):
            b.block[5].weight.copy_(orig[k][0])
            b.block[5].bias.copy_(orig[k][1])
        for k in range(J):
            mask[k] = None
        line = []
        for k in range(J):
            cf = cfg[k]
            S = cf["S"]
            if var == "ind":
                switch(k, cf["c"], torch.ones(len(S)), cf["mu"], cf["mu"])
            else:
                muO, muC, KCC, KOC = seq_stats(k)
                pid = cf["pid"]
                h = pid.shape[0]
                if var == "seq_mu":
                    c, a_s = cf["c"], torch.ones(len(S))
                else:
                    Kpp = KCC[pid[:, :, None], pid[:, None, :]]
                    sdO_k = cf["sdO"]
                    Kdp = KOC[torch.arange(h)[:, None], pid]
                    reg = 1e-4 * KCC.diagonal().mean()
                    c = torch.linalg.solve(Kpp + reg * torch.eye(S_PAR, dtype=Kpp.dtype), Kdp)
                    c = torch.where(cf["c"] != 0, c, torch.zeros_like(c))
                    a_s = KOC.diagonal()[S] / KCC.diagonal()[S].clamp_min(1e-12)
                    if var == "seq_match":  # keep the signal's size: scale = sign x sd_O / sd_C, and rescale c the same way
                        a_s = torch.sign(a_s) * sdO_k[S] / KCC.diagonal()[S].clamp_min(1e-12).sqrt()
                        pv = torch.einsum("hi,hij,hj->h", c, Kpp, c).clamp_min(1e-12).sqrt()
                        c = c * (sdO_k / pv)[:, None]
                    if True:
                        q = torch.quantile(a_s.float(), torch.tensor([.01, .5, .99]))
                        print(f"    {var} block {k}: a quantiles 1/50/99% {q.tolist()}, max {float(a_s.max()):.3g}; "
                              f"c abs max {float(c.abs().max()):.3g} (ind {float(cf['c'].abs().max()):.3g})", flush=True)
                    if var == "seq_a1":
                        a_s = torch.ones(len(S))
                switch(k, c, a_s, muO.float(), muC.float())
            if k in CHECK:
                line.append((k, round(ev(), 4)))
                print(f'  {var} after block {k}: KL {line[-1][1]}', flush=True)
        print(f"{var}: cumulative KL at checkpoints: {line}", flush=True)


if __name__ == "__main__":
    main()
