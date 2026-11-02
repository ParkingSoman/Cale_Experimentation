"""Move 12 of the exploration: is a unit READ? (The essay's definition of clean: parts "built to be read, and built
upon, by its downstream readers".) Lab work.

Signed active read of unit u at (x, p): S_u = sum_r w_ru 1[z_r(x, p') > 0] over its 1x1-type readers r of the next
block (rescaled to unit norm). Read index: lambda(u) = (E_target[S_u] - E_anywhere[S_u]) / sum_r |w_ru|, where
"anywhere" is a uniform position of a random image: the extra positive read the unit gets where it is on. Also the
correlation of h_u with S_u over uniform positions.
usage: s10_read.py <block> <channel> [n_random_units]
Prediction (written before the run): trained units have lambda clearly above 0; random-init units have lambda near 0.
"""
import sys

import numpy as np
import torch

from lib import load_images, load_model, prep, run
from routes import WIDTHS, build_route_use, reader_weights, target_samples

blk, ch = sys.argv[1], int(sys.argv[2])
n_rand = int(sys.argv[3]) if len(sys.argv) > 3 else 8
X, y = load_images()
rng = np.random.default_rng(21)
imgs = np.sort(rng.choice(len(y), 1500, replace=False))
units = [ch] + list(np.random.default_rng(12).choice(sum(WIDTHS[blk]), n_rand, replace=False))


def read_index(m, u, n_pos=1500):
    rw = reader_weights(m, blk, u).cpu().numpy()
    tot = np.abs(rw).sum()
    picks, v = target_samples(m, X, imgs, blk, u, n_pos, np.random.default_rng(22))
    _, Gt = build_route_use(m, X, picks, blk, u)
    G = int(round(np.sqrt(v.shape[1])))
    r2 = np.random.default_rng(23)
    uni = np.stack([r2.choice(imgs, n_pos), r2.integers(0, G, n_pos), r2.integers(0, G, n_pos)], 1)
    _, Gn = build_route_use(m, X, uni, blk, u)
    hu = v.reshape(len(imgs), -1)[np.searchsorted(imgs, uni[:, 0]), uni[:, 1] * G + uni[:, 2]]
    S_t, S_n = Gt.sum(1), Gn.sum(1)
    lam = (S_t.mean() - S_n.mean()) / tot
    corr = np.corrcoef(hu, S_n)[0, 1] if hu.std() > 0 else 0.0
    return lam, corr


def report(m, tag):
    res = {u: read_index(m, u) for u in units}
    print(f"[{tag}] {blk} read index lambda: " + ", ".join(f"{u}: {res[u][0]:+.3f}" for u in units))
    print(f"[{tag}] corr(h_u, S_u): " + ", ".join(f"{u}: {res[u][1]:+.3f}" for u in units))
    print(f"[{tag}] median lambda {np.median([r[0] for r in res.values()]):+.3f}, "
          f"median corr {np.median([r[1] for r in res.values()]):+.3f}")


report(load_model(), "trained")
from lucent.modelzoo import inceptionv1  # noqa: E402
from inception import DEV  # noqa: E402
torch.manual_seed(0)
mr = inceptionv1(pretrained=False).eval().to(DEV)
for p in mr.parameters():
    p.requires_grad_(False)
report(mr, "random init")
