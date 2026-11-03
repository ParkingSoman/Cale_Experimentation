# serves: H-map
"""One pass over the photos, for the map agents. It stores what each channel does on each photo. It does no grouping.

Files under cache/map18/<which>/ (which = read or sealed), n photos by C channels:
   s{s}b{b}_hm  mean h over positions (float16)      s{s}b{b}_hx  largest h (float16)     s{s}b{b}_hp  its position (uint16)
   s{s}b{b}_zm  mean z over positions (float16)      s{s}b{b}_zn  smallest z (float16)    s{s}b{b}_zp  its position (uint16)
   s{s}b{b}_im, _ix, _ip   the same for the inner channel: mean, largest, position
   s0b0_hm, _hx, _hp   the stem. y.npy logits (float32).
   skipstats.npz       per block, per channel, over positions and photos: mean and variance of r and skip, and their covariance.
Position = row * width + column on the block's grid.

Usage: prepare_store.py read|sealed [--smoke]. Size check before a full run: about 1.1 GB for read, .3 GB for sealed."""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from net import STORE, Photos, block_parts, class_names, forward, head, load, stages, stem, WIDTH  # noqa: E402,F401

which = sys.argv[1]
smoke = "--smoke" in sys.argv
BATCH = 64
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
P = Photos(which)
idx = np.arange(len(P))
if smoke:
    idx = idx[:128]
OUT = os.path.join(STORE, which + ("_smoke" if smoke else ""))
os.makedirs(OUT, exist_ok=True)
m = load().to(DEV)
n = len(idx)
arrs = {}


def mm(name, shape, dtype):
    if name not in arrs:
        arrs[name] = np.lib.format.open_memmap(os.path.join(OUT, name + ".npy"), mode="w+", dtype=dtype, shape=shape)
    return arrs[name]


def peak(t):
    """t: batch x C x H x W. Returns mean, max, argmax position, as numpy."""
    f = t.flatten(2)
    v, p = f.max(2)
    return f.mean(2).float().cpu().numpy(), v.float().cpu().numpy(), p.cpu().numpy()


def put(pre, i, t):
    me, mx, po = peak(t)
    C = t.shape[1]
    mm(pre + "m", (n, C), np.float16)[i:i + len(t)] = me
    mm(pre + "x", (n, C), np.float16)[i:i + len(t)] = mx
    mm(pre + "p", (n, C), np.uint16)[i:i + len(t)] = po


stats = {}


def accumulate(key, r, s):
    C = r.shape[1]
    d = stats.setdefault(key, dict(n=0.0, r=torch.zeros(C, dtype=torch.float64), s=torch.zeros(C, dtype=torch.float64),
                                   rr=torch.zeros(C, dtype=torch.float64), ss=torch.zeros(C, dtype=torch.float64),
                                   rs=torch.zeros(C, dtype=torch.float64)))
    fr, fs = r.flatten(2).float().cpu().double(), s.flatten(2).float().cpu().double()
    d["n"] += fr.shape[0] * fr.shape[2]
    d["r"] += fr.sum((0, 2))
    d["s"] += fs.sum((0, 2))
    d["rr"] += (fr * fr).sum((0, 2))
    d["ss"] += (fs * fs).sum((0, 2))
    d["rs"] += (fr * fs).sum((0, 2))


t0 = time.time()
worst = 0.0
with torch.no_grad():
    for i in range(0, n, BATCH):
        sel = idx[i:i + BATCH]
        x = P.tensor(sel).to(DEV)
        h = stem(m, x)
        put("s0b0_h", i, h)
        for s, lay in stages(m).items():
            for b, bl in enumerate(lay):
                p = block_parts(bl, h)
                h = p["h"]
                pre = f"s{s}b{b}_"
                put(pre + "h", i, p["h"])
                put(pre + "i", i, p["inner"])
                f = p["z"].flatten(2)
                mm(pre + "zm", (n, f.shape[1]), np.float16)[i:i + len(sel)] = f.mean(2).float().cpu().numpy()
                v, q = f.min(2)
                mm(pre + "zn", (n, f.shape[1]), np.float16)[i:i + len(sel)] = v.float().cpu().numpy()
                mm(pre + "zp", (n, f.shape[1]), np.uint16)[i:i + len(sel)] = q.cpu().numpy()
                accumulate((s, b), p["r"], p["skip"])
        y = head(m, h)
        mm("y", (n, 1000), np.float32)[i:i + len(sel)] = y.float().cpu().numpy()
        if i == 0:  # the hand-written network must be torchvision's network
            worst = float((y - m(x)).abs().max())
            assert worst < 1e-3, worst
        if (i // BATCH) % 50 == 0:
            print(f"{i}/{n} {time.time() - t0:.0f}s", flush=True)
for a in arrs.values():
    a.flush()
np.savez(os.path.join(OUT, "skipstats.npz"), **{f"s{s}b{b}_{k}": (v if k == "n" else v.numpy()) for (s, b), d in stats.items() for k, v in d.items()})
np.save(os.path.join(OUT, "labels.npy"), P.lab[idx])
print(f"done {n} photos {time.time() - t0:.0f}s; hand-written net against torchvision, largest logit difference {worst:.2e}")
