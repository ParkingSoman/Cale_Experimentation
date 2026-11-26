"""N11: cache the stream at the input of stage 4 (after A's frozen stage 3 and downsample), for every N6 photo set.

Input: the N6 stage-3 cache (gap/cache3.py). Output: results/coarse/cache4/<set>_{x4,logits,ix}.npy.
Check: A run from x4 reproduces A's cached logits (max abs difference printed).
"""
import os
import sys

import numpy as np
import torch

import cc

torch.set_grad_enabled(False)


def main():
    smoke = "--smoke" in sys.argv
    os.makedirs(cc.CACHE4, exist_ok=True)
    base = cc.C.load("base").eval()
    for name in cc.SETS:
        src = cc.C.Cache(name)
        n = 200 if smoke else len(src)
        out = os.path.join(cc.CACHE4, f"{name}_x4.npy")
        x4 = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(n, 1024, 7, 7))
        worst = 0.0
        for a in range(0, n, 100):
            rows = np.arange(a, min(a + 100, n))
            x3, z = src.get(rows)
            h = base.features[6](base.features[5](x3))
            x4[a:a + len(rows)] = h.half().cpu().numpy()
            y = cc.run4(base, h.half().float())
            worst = max(worst, float((y - z).abs().max()))
        x4.flush()
        np.save(os.path.join(cc.CACHE4, f"{name}_logits.npy"), src.z[:n])
        np.save(os.path.join(cc.CACHE4, f"{name}_ix.npy"), src.ix[:n])
        print(f"{name}: {n} photos, A from x4 vs cached logits, max abs diff {worst:.4f}", flush=True)


if __name__ == "__main__":
    main()
