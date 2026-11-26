"""Pair study, open search: which combination of Base neurons rebuilds each Small neuron?

Outputs after GELU (what a neuron sends on), stage-4 block 0 of both models, every photo position.
Photos: pos.photos (2 fit + 2 test per class). Running sums on fit and test rows:
  Base x Base  (4096 x 4096), Base x Small (4096 x 3072), means, Small squares.
Saved to results/pair/grams.npz for greedy.py.
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
    tes = set(te.tolist())
    allix = np.sort(np.concatenate([tr, te]))
    ms, mb = load("small"), load("base")
    G = {}
    ms.features[7][0].block[4].register_forward_hook(lambda m, i, o: G.__setitem__("s", o))
    mb.features[7][0].block[4].register_forward_hook(lambda m, i, o: G.__setitem__("b", o))
    z = lambda *s: torch.zeros(*s, dtype=torch.float64)
    A = {k: dict(n=0, sb=z(4096), ss=z(3072), sbb=z(4096, 4096), sbs=z(4096, 3072), sss=z(3072))
         for k in ("fit", "test")}
    t0 = time.time()
    for k in range(0, len(allix), 50):
        ii = allix[k:k + 50]
        x = ph.tensor(ii).to(DEV)
        ms(x), mb(x)
        b = G["b"].reshape(len(ii), 49, 4096).float()
        s = G["s"].reshape(len(ii), 49, 3072).float()
        for name, mask in (("fit", [i not in tes for i in ii]), ("test", [i in tes for i in ii])):
            mk = torch.tensor(mask, device=DEV)
            if not mk.any():
                continue
            B, S = b[mk].reshape(-1, 4096), s[mk].reshape(-1, 3072)
            a = A[name]
            a["n"] += B.shape[0]
            a["sb"] += B.sum(0).cpu().double()
            a["ss"] += S.sum(0).cpu().double()
            a["sss"] += (S * S).sum(0).cpu().double()
            a["sbb"] += (B.T @ B).cpu().double()
            a["sbs"] += (B.T @ S).cpu().double()
        if (k // 50) % 20 == 0:
            print(f"  {k + len(ii)}/{len(allix)} {time.time() - t0:.0f}s", flush=True)
    out = {}
    for name, a in A.items():
        n = a["n"]
        mb_, ms_ = a["sb"] / n, a["ss"] / n
        out[f"{name}_Cbb"] = (a["sbb"] / n - mb_[:, None] * mb_[None, :]).float().numpy()
        out[f"{name}_Cbs"] = (a["sbs"] / n - mb_[:, None] * ms_[None, :]).float().numpy()
        out[f"{name}_vs"] = (a["sss"] / n - ms_ ** 2).numpy()
        out[f"{name}_mb"], out[f"{name}_ms"], out[f"{name}_n"] = mb_.numpy(), ms_.numpy(), n
    np.savez(os.path.join(OUT, "grams.npz"), **out)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
