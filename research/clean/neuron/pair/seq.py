"""Sequential build: shrink the blocks of stages 3-4 in order, each one solved on the shrunk network's own inputs,
with the ORIGINAL network's stream after that block as the target (so each block also corrects upstream drift).

For block b, on calibration photos (500 fit photos), with the original net O and the shrunk-so-far net C:
  g_c   = hidden outputs of block b in C (from C's own block input x_c)
  target = z_o - x_c   (z_o = O's stream after block b; the block's residual branch should produce this)
  pick S (value on C's covariance, or random), solve Wnew^T = K_SS^-1 Cov(g_S, target), bias to match means.
Then evaluate C end to end on the 1,500 test photos: KL to O and top-1. Keep fraction 0.5 (and 0.25 if time).
"""
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from cache import DEV, load, net, split
from distill import D

torch.set_grad_enabled(False)


def main():
    frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    how = sys.argv[2] if len(sys.argv) > 2 else "value"
    ph = net.Photos("read")
    fit = split(ph.lab)
    cal = np.array(sorted([np.where((ph.lab == c) & fit)[0][9] for c in range(0, 1000, 2)]))
    te = np.load(os.path.join(D, "photos_test.npy"))
    O = load("base")
    Cn = copy.deepcopy(O)
    BO = [blk for s in (5, 7) for blk in O.features[s]]
    BC = [blk for s in (5, 7) for blk in Cn.features[s]]
    rng = np.random.default_rng(0)
    xin = torch.cuda if False else None
    cal_x = [ph.tensor(cal[k:k + 25]) for k in range(0, len(cal), 25)]
    t0 = time.time()
    for b, (bo, bc) in enumerate(zip(BO, BC)):
        H = {}
        h1 = bc.register_forward_pre_hook(lambda mod, inp: H.__setitem__("xc", inp[0]))
        h2 = bc.block[4].register_forward_hook(lambda mod, i, o: H.__setitem__("g", o))
        h3 = bo.register_forward_hook(lambda mod, i, o: H.__setitem__("zo", o))
        hid = bc.block[3].out_features
        d = bc.block[5].out_features
        n, sg, sgg, st, sgt = 0, torch.zeros(hid, dtype=torch.float64), torch.zeros(hid, hid, dtype=torch.float64), \
            torch.zeros(d, dtype=torch.float64), torch.zeros(hid, d, dtype=torch.float64)
        for x in cal_x:
            x = x.to(DEV)
            O(x)
            Cn(x)
            g = H["g"].reshape(-1, hid).float()
            tgt = (H["zo"] - H["xc"]).permute(0, 2, 3, 1).reshape(-1, d).float()
            n += g.shape[0]
            sg += g.sum(0).cpu().double()
            sgg += (g.T @ g).cpu().double()
            st += tgt.sum(0).cpu().double()
            sgt += (g.T @ tgt).cpu().double()
        for h_ in (h1, h2, h3):
            h_.remove()
        mg, mt = sg / n, st / n
        K = sgg / n - mg[:, None] * mg[None, :]
        Cgt = sgt / n - mg[:, None] * mt[None, :]
        ls = bc.layer_scale.detach().view(-1).cpu().double()
        Wb = ls[:, None] * bc.block[5].weight.detach().cpu().double()
        kk = int(round(frac * hid))
        if how == "value":
            S = torch.argsort(-(K.diagonal().clamp_min(0).sqrt() * Wb.norm(dim=0)))[:kk]
        else:
            S = torch.from_numpy(rng.choice(hid, kk, replace=False))
        KSS = K[S][:, S] + 1e-6 * K.diagonal().mean() * torch.eye(kk, dtype=torch.float64)
        Wnew = torch.zeros_like(Wb)
        Wnew[:, S] = torch.linalg.solve(KSS, Cgt[S]).T
        bnew = mt - Wnew @ mg
        bc.block[5].weight.copy_((Wnew / ls[:, None]).float().to(DEV))
        bc.block[5].bias.copy_((bnew / ls).float().to(DEV))
        print(f"  block {b + 1}/{len(BC)} {time.time() - t0:.0f}s", flush=True)
    kl, c, c0, nn_ = 0.0, 0, 0, 0
    for a in range(0, len(te), 50):
        ii = te[a:a + 50]
        x = ph.tensor(ii).to(DEV)
        lab = torch.tensor(ph.lab[ii], device=DEV)
        y0, y1 = O(x), Cn(x)
        kl += float((F.softmax(y0, 1) * (F.log_softmax(y0, 1) - F.log_softmax(y1, 1))).sum())
        c += int((y1.argmax(1) == lab).sum())
        c0 += int((y0.argmax(1) == lab).sum())
        nn_ += len(ii)
    res = dict(frac=frac, how=how, kl=kl / nn_, top1=c / nn_, base_top1=c0 / nn_)
    print(json.dumps(res), flush=True)
    json.dump(res, open(os.path.join(D, f"seq_{how}_{frac}.json"), "w"))


if __name__ == "__main__":
    main()
