# serves: H-map
"""One pass over the photos for what round 2 adds to the store: the negative pole of each inner channel.
Files under cache/map18/<which>/: s{s}b{b}_an  smallest value before the inner ReLU (float16)      s{s}b{b}_ap  its position (uint16).
Size: about 270 MB for read, 80 MB for sealed. Uses the CPU. Usage: prepare_store2.py read|sealed [--smoke]"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from net import STORE, Photos, load, stages, stem  # noqa: E402

which = sys.argv[1]
smoke = "--smoke" in sys.argv
torch.set_num_threads(4)
P = Photos(which)
n = 128 if smoke else len(P)
OUT = os.path.join(STORE, which + ("_smoke" if smoke else ""))
os.makedirs(OUT, exist_ok=True)
m = load()
arrs = {}


def mm(key, shape, dtype):
    if key not in arrs:
        arrs[key] = np.lib.format.open_memmap(os.path.join(OUT, key + ".npy"), mode="w+", dtype=dtype, shape=shape)
    return arrs[key]


t0 = time.time()
with torch.no_grad():
    for i in range(0, n, 64):
        x = P.tensor(np.arange(i, min(i + 64, n)))
        h = stem(m, x)
        for s, lay in stages(m).items():
            for b, bl in enumerate(lay):
                a = bl.bn1(bl.conv1(h))
                f = a.flatten(2)
                v, q = f.min(2)
                mm(f"s{s}b{b}_an", (n, f.shape[1]), np.float16)[i:i + len(x)] = v.numpy()
                mm(f"s{s}b{b}_ap", (n, f.shape[1]), np.uint16)[i:i + len(x)] = q.numpy()
                h = bl(h)
        if i % 3200 == 0:
            print(i, f"{time.time() - t0:.0f}s", flush=True)
for a in arrs.values():
    a.flush()
print("done", n, f"{time.time() - t0:.0f}s")
