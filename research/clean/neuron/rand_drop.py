"""Control for absorb.py: remove RANDOM block-1 inner channels of the same counts (3 and 10, seed 0), so the drop cost on a
parent's own classes can be compared with a removal that has nothing to do with the parent. Saves pooled final stream."""
import os

import numpy as np
import torch

import absorb
from absorb import net, stage4, DEV, BATCH, OUT

torch.set_grad_enabled(False)


def main():
    m = net.load().to(DEV)
    ph = net.Photos("read")
    means = torch.tensor(np.load(os.path.join(OUT, "absorb_cache.npz"))["means"]).to(DEV)
    rng = np.random.default_rng(0)
    used = {j for d in absorb.PARENTS.values() for j in d["map_children"]}
    pool = np.array([j for j in range(512) if j not in used])
    recs = {"rand3": sorted(rng.choice(pool, 3, replace=False).tolist()),
            "rand10": sorted(rng.choice(pool, 10, replace=False).tolist())}
    N = len(ph.lab)
    g = {k: np.zeros((N, 512), np.float32) for k in recs}
    for i in range(0, N, BATCH):
        idx = np.arange(i, min(i + BATCH, N))
        x = ph.tensor(idx).to(DEV)
        h3 = m.layer3(m.layer2(m.layer1(net.stem(m, x))))
        for k, ch in recs.items():
            _, _, hd = stage4(m, h3, drop=ch, means=means)
            g[k][idx] = hd.mean((2, 3)).cpu().numpy()
    np.savez_compressed(os.path.join(OUT, "rand_cache.npz"), **{f"gD_{k}": v for k, v in g.items()},
                        **{f"ch_{k}": np.array(v) for k, v in recs.items()})
    print("saved", recs, flush=True)


if __name__ == "__main__":
    main()
