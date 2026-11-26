"""N5 step 1a: class profile of every hidden neuron in stages 3-4 of ConvNeXt-Base.

One pass over the first P fit photos of each class (default 10). Per neuron:
  peak_cm[c]   mean over class-c photos of the per-photo peak (max over positions) of the GELU output
  peak_mu, peak_sd   mean and std of the per-photo peak over all these photos
  gmean[grp]   mean of the GELU output over positions and photos, on the group's photos ("goal mean"), and on all photos
Saved to results/tree/profile.npz (float16 for the big arrays).
Usage: python profile.py [P]   (P = 1 for a smoke run)
"""
import os
import sys
import time

import numpy as np
import torch

from common import DEV, OUT, blocks, groups, load, net, split

torch.set_grad_enabled(False)


def main():
    P = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    ph = net.Photos("read")
    fit = split(ph.lab)
    idx = []
    for c in range(1000):
        idx += np.where((ph.lab == c) & fit)[0][:P].tolist()
    idx = np.array(sorted(idx))
    lab = ph.lab[idx]
    G = groups()
    gid = np.full(1000, -1)
    names = list(G)
    for k, n in enumerate(names):
        gid[G[n]] = k
    m = load("base")
    BL = blocks(m)
    H = [blk.block[3].out_features for blk in BL]
    pk_sum = [torch.zeros(1000, h) for h in H]
    s1 = [torch.zeros(h, dtype=torch.float64) for h in H]
    s2 = [torch.zeros(h, dtype=torch.float64) for h in H]
    gm = [torch.zeros(len(names) + 1, h, dtype=torch.float64) for h in H]
    gn = np.zeros(len(names) + 1)
    cur = {}
    for g, blk in enumerate(BL):
        blk.block[4].register_forward_hook(lambda mod, i, o, g=g: cur.__setitem__(g, o))
    t0 = time.time()
    B = 50
    for k in range(0, len(idx), B):
        ii = idx[k:k + B]
        m(ph.tensor(ii).to(DEV))
        lb = torch.from_numpy(ph.lab[ii])
        gg = torch.from_numpy(gid[ph.lab[ii]])
        for g in range(len(BL)):
            o = cur[g]
            pk = o.amax((1, 2)).float().cpu()
            sm = o.mean((1, 2)).float().cpu().double()
            pk_sum[g].index_add_(0, lb, pk)
            s1[g] += pk.double().sum(0)
            s2[g] += pk.double().pow(2).sum(0)
            gm[g][-1] += sm.sum(0)
            for j in range(len(names)):
                w = gg == j
                if w.any():
                    gm[g][j] += sm[w].sum(0)
        for j in range(len(names)):
            gn[j] += int((gg == j).sum())
        gn[-1] += len(ii)
        if (k // B) % 20 == 0:
            print(f"  {k + len(ii)}/{len(idx)} {time.time() - t0:.0f}s", flush=True)
    n = len(idx)
    cnt = np.bincount(lab, minlength=1000).astype(np.float32)
    out = {}
    for g in range(len(BL)):
        out[f"cm{g}"] = (pk_sum[g] / torch.from_numpy(cnt)[:, None]).numpy().astype(np.float16)
        mu = s1[g] / n
        out[f"mu{g}"] = mu.float().numpy()
        out[f"sd{g}"] = (s2[g] / n - mu ** 2).clamp_min(1e-12).sqrt().float().numpy()
        out[f"gm{g}"] = (gm[g] / torch.from_numpy(gn)[:, None]).float().numpy()
    out["groups"] = np.array(names + ["all"])
    np.savez(os.path.join(OUT, f"profile_P{P}.npz"), **out)
    print(f"done {time.time() - t0:.0f}s, photos {n}", flush=True)


if __name__ == "__main__":
    main()
