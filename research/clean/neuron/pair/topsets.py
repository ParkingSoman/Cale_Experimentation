"""Pair study, firing sets: is a Small neuron the UNION of a few Base neurons' firings?

Stage-4 block 0, outputs after GELU, all positions of the pos.photos photos (2 fit + 2 test per class).
For every neuron of both models: its top 1% rows (photo x position), kept separately on fit and on test rows.
Saved to results/pair/topsets.npz (row ids, int32). cover.py does the analysis.
"""
import os
import time

import numpy as np
import torch

from cache import DEV, OUT, load, net
from pos import photos

torch.set_grad_enabled(False)


def main():
    ph = net.Photos("read")
    tr, te = photos(ph)
    out = {}
    ms, mb = load("small"), load("base")
    G = {}
    ms.features[7][0].block[4].register_forward_hook(lambda m, i, o: G.__setitem__("s", o))
    mb.features[7][0].block[4].register_forward_hook(lambda m, i, o: G.__setitem__("b", o))
    for name, ix in (("fit", tr), ("test", te)):
        K = int(len(ix) * 49 * 0.01)
        tv = torch.full((7168, K), -1e9, device=DEV)
        ti = torch.zeros((7168, K), dtype=torch.int32, device=DEV)
        t0 = time.time()
        for k in range(0, len(ix), 50):
            ii = ix[k:k + 50]
            x = ph.tensor(ii).to(DEV)
            ms(x), mb(x)
            a = torch.cat([G["s"].reshape(-1, 3072), G["b"].reshape(-1, 4096)], 1).float().T  # (7168, rows)
            rid = torch.arange(k * 49, k * 49 + a.shape[1], device=DEV, dtype=torch.int32)
            v = torch.cat([tv, a], 1)
            i = torch.cat([ti, rid.expand(7168, -1)], 1)
            o = torch.topk(v, K, dim=1).indices
            tv, ti = torch.gather(v, 1, o), torch.gather(i, 1, o)
            if (k // 50) % 10 == 0:
                print(f"  {name} {k + len(ii)}/{len(ix)} {time.time() - t0:.0f}s", flush=True)
        out[f"{name}_ids"] = ti.cpu().numpy()
        out[f"{name}_vals"] = tv.cpu().numpy().astype(np.float16)
        out[f"{name}_nrows"] = len(ix) * 49
    out["fit_photos"], out["test_photos"] = tr, te
    np.savez(os.path.join(OUT, "topsets.npz"), **out)


if __name__ == "__main__":
    main()
