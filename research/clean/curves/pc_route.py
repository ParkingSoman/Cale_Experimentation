# serves: H-route
"""Part C measurement of Hypothesis H-route (COARSE-THEORY.md A.3): packing index pi and read index lambda of
Definition D45 for units of InceptionV1 blocks 3b, 4c, 4e, trained and at a random initialisation, and planted
two-job units. Estimator as written in the Part C row of H-route before the run. Uses the exploration helpers in
research/clean/explore (routes.py).
Thresholds fixed before the run: (1) trained median pi <= 1/4 x planted median pi; (2) trained share lambda > 0 >= 9/10;
(3) random-init share lambda > 0 <= 3/4.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "explore"))
from lib import load_images, load_model  # noqa: E402
from routes import WIDTHS, build_route_use, packing_index, reader_weights, target_samples  # noqa: E402

X, y = load_images()
imgs = np.sort(np.random.default_rng(31).choice(len(y), 1500, replace=False))
BLOCKS_ = ["3b", "4c", "4e"]
units = {b: list(np.random.default_rng(32 + i).choice(sum(WIDTHS[b]), 12, replace=False)) for i, b in enumerate(BLOCKS_)}


def measure(m, blk, u, n_pos=1500):
    picks, v = target_samples(m, X, imgs, blk, u, n_pos, np.random.default_rng(33))
    if picks is None:
        return None
    U, Gt = build_route_use(m, X, picks, blk, u)
    pi = packing_index(U, Gt)[0]
    G = int(round(np.sqrt(v.shape[1])))
    r2 = np.random.default_rng(34)
    uni = np.stack([r2.choice(imgs, n_pos), r2.integers(0, G, n_pos), r2.integers(0, G, n_pos)], 1)
    _, Gn = build_route_use(m, X, uni, blk, u)
    T = np.abs(reader_weights(m, blk, u).cpu().numpy()).sum()
    lam = float((Gt.sum(1).mean() - Gn.sum(1).mean()) / T)
    return dict(pi=float(pi), lam=lam, Gt=Gt)


def run_net(m, tag):
    out = {}
    for blk in BLOCKS_:
        res = {}
        for u in units[blk]:
            r = measure(m, blk, int(u))
            if r is not None:
                res[int(u)] = r
        planted = []
        ks = list(res)
        for a, b in zip(ks[:-1], ks[1:]):
            G1, G2 = res[a]["Gt"], res[b]["Gt"]
            Uv = np.zeros((len(G1) + len(G2), 2)); Uv[:len(G1), 0] = 1; Uv[len(G1):, 1] = 1
            planted.append(float(packing_index(Uv, np.concatenate([G1, G2]))[0]))
        out[blk] = dict(pi={u: r["pi"] for u, r in res.items()}, lam={u: r["lam"] for u, r in res.items()},
                        planted=planted)
        print(f"[{tag}] {blk}: live {len(res)}/12; median pi {np.median(list(out[blk]['pi'].values())):.4f}; "
              f"planted median {np.median(planted):.4f}; share lambda > 0 "
              f"{np.mean([l > 0 for l in out[blk]['lam'].values()]):.3f}; median lambda "
              f"{np.median(list(out[blk]['lam'].values())):+.4f}", flush=True)
    allpi = [p for b in BLOCKS_ for p in out[b]["pi"].values()]
    allpl = [p for b in BLOCKS_ for p in out[b]["planted"]]
    alllam = [l for b in BLOCKS_ for l in out[b]["lam"].values()]
    summ = dict(median_pi=float(np.median(allpi)), median_planted=float(np.median(allpl)),
                share_lam_pos=float(np.mean([l > 0 for l in alllam])), n_units=len(allpi))
    print(f"[{tag}] all blocks: {summ}", flush=True)
    return out, summ


res = {}
ONLY_RANDOM = len(sys.argv) > 1 and sys.argv[1] == "random"
if ONLY_RANDOM:  # the trained half finished in an earlier run; its summary is passed back in
    res["trained"] = (None, json.loads(sys.argv[2]))
else:
    res["trained"] = run_net(load_model(), "trained")
from lucent.modelzoo import inceptionv1  # noqa: E402
from inception import DEV  # noqa: E402
torch.manual_seed(0)
mr = inceptionv1(pretrained=False).eval().to(DEV)
for p in mr.parameters():
    p.requires_grad_(False)
res["random"] = run_net(mr, "random init")
t, r = res["trained"][1], res["random"][1]
c1 = t["median_pi"] <= 0.25 * t["median_planted"]
c2 = t["share_lam_pos"] >= 0.9
c3 = r["share_lam_pos"] <= 0.75
print(f"H-route: (1) {c1} ({t['median_pi']:.4f} vs 1/4 of {t['median_planted']:.4f}); (2) {c2} "
      f"({t['share_lam_pos']:.3f}); (3) {c3} ({r['share_lam_pos']:.3f})")
json.dump({k: {"blocks": v[0], "summary": v[1]} for k, v in res.items()},
          open(os.path.join(HERE, "results_pc_route" + ("_random" if ONLY_RANDOM else "") + ".json"), "w"), indent=1,
          default=str)
