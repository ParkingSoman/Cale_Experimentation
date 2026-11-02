"""Move 11 of the exploration: the scale of the packing index. Lab work.

(a) planted two-job units: for a pair of real units (u1, u2) of one block, a virtual unit that is u1 on u1's target and
    u2 on u2's target. Its routes are "u1" and "u2" and its uses are the real gated reads of u1 and of u2. Its pi is
    what the index gives a unit whose two jobs feed two different real reader sets.
(b) the packing index of single units of one block: trained network, and the same units in a randomly initialised
    InceptionV1.
usage: s9_scale.py <block> <channel> [n_random_units]
Prediction (written before the run): planted pairs score well above single trained units (at least 5 times their
median); random-init units score higher than trained units (median).
"""
import sys

import numpy as np
import torch

from lib import load_images, load_model
from routes import WIDTHS, build_route_use, packing_index, target_samples

blk, ch = sys.argv[1], int(sys.argv[2])
n_rand = int(sys.argv[3]) if len(sys.argv) > 3 else 8
X, y = load_images()
rng = np.random.default_rng(11)
imgs = np.sort(rng.choice(len(y), 1500, replace=False))
units = [ch] + list(np.random.default_rng(12).choice(sum(WIDTHS[blk]), n_rand, replace=False))


def unit_stats(m, u, n_pos=1500):
    picks, v = target_samples(m, X, imgs, blk, u, n_pos, np.random.default_rng(13))
    U, Gu = build_route_use(m, X, picks, blk, u)
    return U, Gu, v.sum()


def report(m, tag):
    S = {u: unit_stats(m, u) for u in units}
    pis = {u: packing_index(S[u][0], S[u][1])[0] for u in units}
    print(f"[{tag}] single units of {blk}: " + ", ".join(f"{u}: {pis[u]:.3f}" for u in units))
    print(f"[{tag}] median single-unit pi {np.median(list(pis.values())):.3f}")
    planted = []
    for u2 in units[1:]:
        G1, G2 = S[ch][1], S[u2][1]
        n1, n2 = len(G1), len(G2)
        U = np.zeros((n1 + n2, 2))
        U[:n1, 0], U[n1:, 1] = 1, 1  # route = which job
        Gu = np.concatenate([G1, np.pad(G2, ((0, 0), (0, 0)))])
        # the two jobs have different reader weights; put both on one reader axis set: reads of u1 then reads of u2
        Gu = np.concatenate([np.concatenate([G1, np.zeros_like(G1)], 1), np.concatenate([np.zeros_like(G2), G2], 1)])
        Gsame = np.concatenate([G1, G2])  # same reader axes (the same readers, each with its own weights)
        planted.append((packing_index(U, Gsame)[0], packing_index(U, Gu)[0]))
    p_same = [a for a, _ in planted]
    print(f"[{tag}] planted pairs ({blk}:{ch} with each other unit), uses on the shared reader axes: "
          + ", ".join(f"{a:.3f}" for a in p_same))
    print(f"[{tag}] planted median {np.median(p_same):.3f}; ratio to the single-unit median "
          f"{np.median(p_same) / max(np.median(list(pis.values())), 1e-9):.1f}")
    return pis


m = load_model()
report(m, "trained")
from lucent.modelzoo import inceptionv1  # noqa: E402
from inception import DEV  # noqa: E402
torch.manual_seed(0)
mr = inceptionv1(pretrained=False).eval().to(DEV)
for p in mr.parameters():
    p.requires_grad_(False)
report(mr, "random init")
