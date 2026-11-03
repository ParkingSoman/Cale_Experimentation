# serves: H-map
"""LADDER 45 said a last-layer channel is fed "mostly by its own skip", and card 3-34 said the last block is about .88 skip.
The job table says the skip holds about .01 of the variance at stage 4, block 1. This compares the two readings on the same
photos. For each channel's top 32 photos, at its own peak in block 1:
   z        z at block 1 at the peak (what the channel reaches)
   skip     h of block 0 at that position (what the skip gives there)
   r        the residual path at that position (z minus skip)
   inxgrad  own skip's share of the positive input x gradient on h of block 0: the older kind of number
PREDICTION, written before the run: at s4, skip is under .2 of z, and own skip's share of input x gradient is well above that,
so the older number overstates the skip because it leaves out the constants that a batch norm adds. Stages 1 to 3: skip .3 to .7 of z."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from net import STORE, Photos, block_parts, load, stages, upto  # noqa: E402

READ = os.path.join(STORE, "read")
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
m = load().to(DEV)
P = Photos("read")
rng = np.random.default_rng(0)
for s, lay in stages(m).items():
    hx = np.load(os.path.join(READ, f"s{s}b1_hx.npy"), mmap_mode="r")
    hp = np.load(os.path.join(READ, f"s{s}b1_hp.npy"), mmap_mode="r")
    C = hx.shape[1]
    rows = []
    for c in rng.choice(C, min(C, 40), replace=False):
        idx = np.argsort(-np.asarray(hx[:, c]))[:32]
        pos = torch.as_tensor(np.asarray(hp[idx, c]).astype(np.int64), device=DEV)
        x = P.tensor(idx).to(DEV)
        with torch.no_grad():
            h_in = upto(m, x, s)
            h0 = block_parts(lay[0], h_in)["h"]
        h0 = h0.detach().requires_grad_(True)
        p1 = block_parts(lay[1], h0)
        k = torch.arange(len(idx), device=DEV)
        z = p1["z"][:, c].flatten(1)[k, pos]
        skip = p1["skip"][:, c].flatten(1)[k, pos]
        g, = torch.autograd.grad(z.sum(), [h0])
        at = (h0 * g).sum((2, 3)).mean(0)
        rows.append((float(z.mean()), float(skip.mean()), float(at[c] / at[at > 0].sum()), float(at.sum())))
    r = np.array(rows)
    print(f"s{s}: median z at the peak {np.median(r[:, 0]):.2f}, skip there {np.median(r[:, 1]):.2f}, skip / z {np.median(r[:, 1] / r[:, 0]):.2f}; "
          f"own skip's share of positive input x gradient {np.median(r[:, 2]):.2f}; sum of input x gradient {np.median(r[:, 3]):.2f}", flush=True)
