"""Move 9 of the exploration: split a unit's target by the network's OWN build pattern (no class labels), then ask
whether the use differs between the parts. And the line-up number: the distance correlation between build and use
on the target (bias-corrected; about 0 when build and use are independent). Lab work.

usage: s7_split.py <block> <channel> [n_images] [n_pos]
Prediction for 4e:55 (written before the run): two clusters of the build vectors (cosine k-means) come out close to
"animal faces" and "vehicles and other objects"; their mean use vectors differ (cosine well below the within-cluster
similarity); the build-use distance correlation is clearly above 0 (above its permutation null).
"""
import os
import sys

import numpy as np

from lib import OUT, class_names, load_images, load_model
from roles import build_and_use, dcor_u, route_index, sample_target, unit_rows

blk, ch = sys.argv[1], int(sys.argv[2])
n_img = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
n_pos = int(sys.argv[4]) if len(sys.argv) > 4 else 600
m = load_model()
X, y = load_images()
names = class_names()
rng = np.random.default_rng(5)
imgs = np.sort(rng.choice(len(y), n_img, replace=False))
picks, G = sample_target(m, X, imgs, blk, ch, n_pos, rng)
use = sys.argv[5] if len(sys.argv) > 5 else "next"
U, D = build_and_use(m, X, picks, blk, ch, use=use)
print("use measured at:", use)
Un, Dn = unit_rows(U), unit_rows(D)

# cosine k-means, k = 2, on the build vectors
best = None
for rs in range(10):
    r2 = np.random.default_rng(rs)
    C = Un[r2.choice(len(Un), 2, replace=False)]
    for _ in range(50):
        lab = np.argmax(Un @ C.T, 1)
        C = np.stack([unit_rows(Un[lab == k].mean(0, keepdims=True))[0] if (lab == k).any() else C[k] for k in range(2)])
    score = (Un * C[lab]).sum()
    if best is None or score > best[0]:
        best = (score, lab, C)
_, lab, C = best
print(f"{blk}:{ch}: {n_pos} target positions from {n_img} images; cluster sizes {np.bincount(lab, minlength=2)}")
for k in range(2):
    ix = np.where(lab == k)[0]
    cls = [names[y[picks[i][0]]].split(",")[0] for i in ix]
    animal = np.mean([y[picks[i][0]] < 398 for i in ix])
    vals, cnt = np.unique(cls, return_counts=True)
    top = [f"{v} {c}" for v, c in sorted(zip(vals, cnt), key=lambda t: -t[1])[:8]]
    lead = np.argsort(-C[k])[:5]
    print(f"  cluster {k}: animal share {animal:.2f}; leading builders (units below) {list(lead)}; classes {top}")
mD = [unit_rows(Dn[lab == k].mean(0, keepdims=True))[0] for k in range(2)]
within = [np.mean(Dn[lab == k] @ mD[k]) for k in range(2)]
print(f"use: cosine between the clusters' mean use directions {float(mD[0] @ mD[1]):.3f}; "
      f"mean cosine of a use vector to its own cluster's mean {within[0]:.3f}, {within[1]:.3f}")
dc = dcor_u(Un, Dn)
null = [dcor_u(Un, Dn[np.random.default_rng(s).permutation(len(Dn))]) for s in range(20)]
print(f"line-up: distance correlation of build and use {dc:.4f}; permutation null mean {np.mean(null):.4f}, "
      f"max {np.max(null):.4f}")
pi, sizes = route_index(U, D)
pi_null = [route_index(U, D[np.random.default_rng(s).permutation(len(D))])[0] for s in range(20)]
print(f"route index (lead builder routes {sizes}): pi {pi:.3f}; shuffled-use null mean {np.mean(pi_null):.3f}, "
      f"max {np.max(pi_null):.3f}")
np.savez(os.path.join(OUT, f"{blk}_{ch}_{use}_roles.npz"), U=U, D=D, lab=lab, picks=np.array(picks), dcor=dc, null=null)
