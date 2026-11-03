# serves: H-map
"""What feeds each stream channel, looking through the skip. It stores raw numbers and picks no parent.

For stream channel c of stage s, and a band of 32 reading photos:
   target   z of c at stage s, block 1, at c's own peak position on each photo (the negative peak for band neg)
   par      input x gradient of the target on the input of stage s, summed over positions, mean over photos: one number per
            channel of the stage below. It counts every route: downsample, block 0's and block 1's residual paths.
   lat      the same on h of stage s, block 0: one number per channel of this stage. Entry c is its own skip.
Bands: top = strongest 32 by largest h. mid = 32 spread over the 60-70th percentile of the photos where c is on. neg = 32
with the most negative z. Files: cache/map18/edges/s{s}_{band}_{par,lat}.npy, shape C x C_below and C x C.
A ReLU network is linear between its kinks, so input x gradient plus a constant equals the target.

Usage: prepare_edges.py [--smoke]. Needs cache/map18/read from prepare_store.py."""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from net import STORE, WIDTH, Photos, block_parts, load, stages, upto  # noqa: E402

smoke = "--smoke" in sys.argv
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
READ = os.path.join(STORE, "read")
OUT = os.path.join(STORE, "edges" + ("_smoke" if smoke else ""))
os.makedirs(OUT, exist_ok=True)
P = Photos("read")
m = load().to(DEV)
N = 32


def band_photos(hx, zn, band):
    if band == "top":
        return np.argsort(-hx)[:N]
    if band == "neg":
        return np.argsort(zn)[:N]
    on = np.where(hx > 1e-3)[0]
    if len(on) < N:
        return np.argsort(-hx)[:N]
    o = on[np.argsort(-hx[on])]
    pick = o[int(.30 * len(o)):max(int(.40 * len(o)), int(.30 * len(o)) + N)]
    return pick[np.linspace(0, len(pick) - 1, N).astype(int)]


t0 = time.time()
for s, lay in stages(m).items():
    C, Cb = WIDTH[s], WIDTH[s - 1]
    g = int(round(224 / {1: 4, 2: 8, 3: 16, 4: 32}[s]))
    hx = np.load(os.path.join(READ, f"s{s}b1_hx.npy"), mmap_mode="r")
    hp = np.load(os.path.join(READ, f"s{s}b1_hp.npy"), mmap_mode="r")
    zn = np.load(os.path.join(READ, f"s{s}b1_zn.npy"), mmap_mode="r")
    zp = np.load(os.path.join(READ, f"s{s}b1_zp.npy"), mmap_mode="r")
    for band in ("top", "mid", "neg"):
        par = np.zeros((C, Cb), np.float32)
        lat = np.zeros((C, C), np.float32)
        chans = range(3 if smoke else C)
        for c in chans:
            idx = band_photos(np.asarray(hx[:, c]), np.asarray(zn[:, c]), band)
            pos = np.asarray(zp[idx, c] if band == "neg" else hp[idx, c]).astype(int)
            x = P.tensor(idx).to(DEV)
            with torch.no_grad():
                h_in = upto(m, x, s)
            h_in = h_in.detach().requires_grad_(True)
            p0 = block_parts(lay[0], h_in)
            h0 = p0["h"]
            p1 = block_parts(lay[1], h0)
            z = p1["z"][:, c].flatten(1)
            tgt = z[torch.arange(len(idx)), torch.as_tensor(pos, device=DEV)].sum()
            g_in, g_0 = torch.autograd.grad(tgt, [h_in, h0])
            par[c] = ((h_in * g_in).sum((2, 3)).mean(0)).detach().cpu().numpy()
            lat[c] = ((h0 * g_0).sum((2, 3)).mean(0)).detach().cpu().numpy()
        np.save(os.path.join(OUT, f"s{s}_{band}_par.npy"), par)
        np.save(os.path.join(OUT, f"s{s}_{band}_lat.npy"), lat)
        print(f"stage {s} {band} {time.time() - t0:.0f}s", flush=True)
print("done")
