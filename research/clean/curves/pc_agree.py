# serves: H-agree
"""Part C measurement of Hypothesis H-agree (COARSE-THEORY.md A.3): the excess read xi of Definition D45 (item 2')
for 12 fresh units per block of InceptionV1 (3b, 4c, 4e), trained and random init. Estimator as in the Part C row.
Thresholds fixed before the run: (1) trained share xi > 0 >= 9/10; (2) random share xi > 0 <= 3/4;
(3) trained median |xi| >= 5 x random median |xi|.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "explore"))
from lib import load_images, load_model  # noqa: E402
from routes import WIDTHS, read_detail, target_samples  # noqa: E402

X, y = load_images()
imgs = np.sort(np.random.default_rng(51).choice(len(y), 1200, replace=False))
BL = ["3b", "4c", "4e"]
units = {b: [int(u) for u in np.random.default_rng(52 + i).choice(sum(WIDTHS[b]), 12, replace=False)]
         for i, b in enumerate(BL)}
N_POS = 1000


def xi(m, blk, u):
    picks, v = target_samples(m, X, imgs, blk, u, N_POS, np.random.default_rng(53))
    if picks is None:
        return None
    G = int(round(np.sqrt(v.shape[1])))
    r2 = np.random.default_rng(54)
    uni = lambda: np.stack([r2.choice(imgs, N_POS), r2.integers(0, G, N_POS), r2.integers(0, G, N_POS)], 1)
    Zt, Ht, w = read_detail(m, X, picks, blk, u)
    Zn, Hn, _ = read_detail(m, X, uni(), blk, u)
    Zi, Hi, _ = read_detail(m, X, uni(), blk, u)
    T = np.abs(w).sum()
    S = lambda Z: ((Z > 0) * w[None, :]).sum(1)
    lam = (S(Zt).mean() - S(Zn).mean()) / T
    R = Zi - Hi[:, None] * w[None, :]
    lam_ind = (S(Ht[:, None] * w[None, :] + R).mean() - S(Hn[:, None] * w[None, :] + R).mean()) / T
    return float(lam), float(lam_ind), float(lam - lam_ind)


def run_net(m, tag):
    out = {}
    for b in BL:
        out[b] = {}
        for u in units[b]:
            r = xi(m, b, u)
            if r is not None:
                out[b][u] = r
        xs = np.array([r[2] for r in out[b].values()])
        print(f"[{tag}] {b}: live {len(xs)}/12; share xi > 0 {np.mean(xs > 0):.3f}; median xi {np.median(xs):+.4f}; "
              f"median |xi| {np.median(np.abs(xs)):.4f}", flush=True)
    xs = np.array([r[2] for b in BL for r in out[b].values()])
    return out, dict(share_pos=float(np.mean(xs > 0)), med_abs=float(np.median(np.abs(xs))), n=len(xs))


res = {"trained": run_net(load_model(), "trained")}
from lucent.modelzoo import inceptionv1  # noqa: E402
from inception import DEV  # noqa: E402
torch.manual_seed(0)
mr = inceptionv1(pretrained=False).eval().to(DEV)
for p in mr.parameters():
    p.requires_grad_(False)
res["random"] = run_net(mr, "random init")
t, r = res["trained"][1], res["random"][1]
print(f"H-agree: trained {t}; random {r}")
print(f"H-agree: (1) {t['share_pos'] >= 0.9}; (2) {r['share_pos'] <= 0.75}; (3) {t['med_abs'] >= 5 * r['med_abs']}")
json.dump({k: {"units": {b: v[0][b] for b in BL}, "summary": v[1]} for k, v in res.items()},
          open(os.path.join(HERE, "results_pc_agree.json"), "w"), indent=1)
