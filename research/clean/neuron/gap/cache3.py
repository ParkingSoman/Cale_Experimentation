"""N6 step 1a: cache the stream at the input of stage 3 (fp16) and Base's logits, for train / trainflip / held / cal / test.

Sizes: 512 x 14 x 14 fp16 = 200 KB per photo. About 7 GB in all (results/gap/cache/, ignored by git).
--n N: smoke mode, the first N photos of each set only, into cache_smoke/.
Sanity: Base's logits from the cached stream (tail) equal the full-model logits (max difference printed).
"""
import argparse
import os
import time

import numpy as np
import torch

import common as C

torch.set_grad_enabled(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    if a.n:
        C.CACHE = os.path.join(C.OUT, "cache_smoke")
    os.makedirs(C.CACHE, exist_ok=True)
    ph = C.net.Photos("read")
    sets = C.photo_sets(ph.lab)
    m = C.load("base")
    head = torch.nn.Sequential(*[m.features[i] for i in range(5)])
    for name, flip in (("test", False), ("cal", False), ("held", False), ("train", False), ("trainflip", True)):
        ix = sets["train" if name == "trainflip" else name]
        if a.n:
            ix = ix[:a.n]
        X = np.lib.format.open_memmap(os.path.join(C.CACHE, f"{name}_x3.npy"), "w+", np.float16, (len(ix), 512, 14, 14))
        Z = np.zeros((len(ix), 1000), np.float32)
        t0, worst = time.time(), 0.0
        for s in range(0, len(ix), 50):
            x = ph.tensor(ix[s:s + 50])
            if flip:
                x = torch.flip(x, dims=[3])
            x = x.to(C.DEV)
            x3 = head(x)
            z = C.tail(m, x3)
            X[s:s + 50] = x3.half().cpu().numpy()
            Z[s:s + 50] = z.cpu().numpy()
            if s == 0:
                worst = float((C.tail(m, x3.half().float()) - m(x)).abs().max())
            if (s // 50) % 40 == 0:
                print(f"{name} {s + 50}/{len(ix)} {time.time() - t0:.0f}s", flush=True)
        X.flush()
        del X
        np.save(os.path.join(C.CACHE, f"{name}_logits.npy"), Z)
        np.save(os.path.join(C.CACHE, f"{name}_ix.npy"), ix)
        print(f"{name}: {len(ix)} photos, {time.time() - t0:.0f}s, fp16 cache vs full logits max diff {worst:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
