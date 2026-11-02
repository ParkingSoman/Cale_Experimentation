"""Move 13 of the exploration: the excess read xi(u) = lambda(u) - lambda_ind(u) (comment (f) of D45): the read index
minus the read index computed with the rest of each reader's input drawn independently of the input. Lab work.

usage: s11_excess.py <block> [n_units] [n_pos]
Prediction (written before the run): trained units have xi > 0 for at least 9 in 10; random-init units have xi of both
signs, positive for at most 3 in 4, and their median |xi| is far below the trained median.
"""
import sys

import numpy as np
import torch

from lib import load_images, load_model
from routes import WIDTHS, read_detail, target_samples

blk = sys.argv[1]
n_units = int(sys.argv[2]) if len(sys.argv) > 2 else 10
n_pos = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
X, y = load_images()
imgs = np.sort(np.random.default_rng(41).choice(len(y), 1200, replace=False))
units = list(np.random.default_rng(42).choice(sum(WIDTHS[blk]), n_units, replace=False))


def xi(m, u):
    picks, v = target_samples(m, X, imgs, blk, u, n_pos, np.random.default_rng(43))
    if picks is None:
        return None
    G = int(round(np.sqrt(v.shape[1])))
    r2 = np.random.default_rng(44)
    uni = lambda: np.stack([r2.choice(imgs, n_pos), r2.integers(0, G, n_pos), r2.integers(0, G, n_pos)], 1)
    Zt, Ht, w = read_detail(m, X, picks, blk, u)
    Zn, Hn, _ = read_detail(m, X, uni(), blk, u)
    Zi, Hi, _ = read_detail(m, X, uni(), blk, u)  # an independent sample for the rest
    T = np.abs(w).sum()
    S = lambda Z: ((Z > 0) * w[None, :]).sum(1)
    lam = (S(Zt).mean() - S(Zn).mean()) / T
    Rt = Zi - Hi[:, None] * w[None, :]  # rest of the readers' input at independent inputs
    # the rest comes from a third, independent sample, for both the target and the anywhere term
    lam_ind = (S(Ht[:, None] * w[None, :] + Rt).mean() - S(Hn[:, None] * w[None, :] + Rt).mean()) / T
    return lam, lam_ind, lam - lam_ind


def report(m, tag):
    res = {int(u): xi(m, int(u)) for u in units}
    res = {u: r for u, r in res.items() if r is not None}
    xs = np.array([r[2] for r in res.values()])
    print(f"[{tag}] {blk}: live {len(res)}/{len(units)}; lambda median {np.median([r[0] for r in res.values()]):+.4f}; "
          f"lambda_ind median {np.median([r[1] for r in res.values()]):+.4f}; xi median {np.median(xs):+.4f}; "
          f"share xi > 0 {np.mean(xs > 0):.2f}; median |xi| {np.median(np.abs(xs)):.4f}", flush=True)


report(load_model(), "trained")
from lucent.modelzoo import inceptionv1  # noqa: E402
from inception import DEV  # noqa: E402
torch.manual_seed(0)
mr = inceptionv1(pretrained=False).eval().to(DEV)
for p in mr.parameters():
    p.requires_grad_(False)
report(mr, "random init")
