"""Moves 9 and 10 of the exploration: split a unit's target by its lead builder (the network's own route, no class
labels), show the top images of each large route, and compare the routes' mean uses (gated reads of the next block).
The packing index pi is 0 when all routes feed the same readers (up to strength). Lab work.

usage: s8_routes.py <block> <channel> [n_images] [n_pos]
Prediction for 4e:55 (written before the run): its two largest routes are 4d:493 (pointed faces) and a vehicle or
broad-band builder; their mean uses differ (cosine well below 1), and pi is clearly above its shuffled-route null.
Prediction for a curve unit of 3b (379): routes are curve-fragment builders of nearby orientations, all feeding the
same readers, so pi is near its null.
"""
import os
import sys

import numpy as np

from lib import OUT, class_names, load_images, load_model, sheet
from routes import build_route_use, packing_index, target_samples

blk, ch = sys.argv[1], int(sys.argv[2])
n_img = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
n_pos = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
m = load_model()
X, y = load_images()
names = class_names()
rng = np.random.default_rng(7)
imgs = np.sort(rng.choice(len(y), n_img, replace=False))
picks, v = target_samples(m, X, imgs, blk, ch, n_pos, rng)
U, Gu = build_route_use(m, X, picks, blk, ch)
pi, sizes, means = packing_index(U, Gu)
null = []
for s in range(20):
    perm = np.random.default_rng(100 + s).permutation(len(U))
    null.append(packing_index(U[perm], Gu)[0])
print(f"{blk}:{ch}: {n_pos} target positions from {n_img} images; unit on in {(v.max(1) > 0).mean():.3f} of images")
print(f"packing index pi {pi:.3f}; shuffled-route null mean {np.mean(null):.3f}, max {np.max(null):.3f}")
big = sorted(sizes, key=lambda g: -sizes[g])[:4]
print("largest routes (lead builder: share of target):", [(g, round(sizes[g] / n_pos, 3)) for g in big])
for g1 in big:
    row = []
    for g2 in big:
        a, b = means[g1], means[g2]
        row.append(round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)), 2))
    print(f"  route {g1}: |mean use| {np.linalg.norm(means[g1]):.3f}; cosines to the big routes {row}; "
          f"top readers {list(np.argsort(-means[g1])[:5])}")
lead = np.argmax(U, 1)
ims, caps, boxes = [], [], []
G = int(round(np.sqrt(v.shape[1])))
st = 224 / G
for g in big:
    ix = np.where(lead == g)[0]
    ix = ix[np.argsort(-U[ix].max(1))][:6]
    for i in ix:
        im, r, c = picks[i]
        ims.append(X[im]); caps.append(f"route {g}: {names[y[im]]}")
        boxes.append(((c + .5) * st - 36, (r + .5) * st - 36, (c + .5) * st + 36, (r + .5) * st + 36))
print(sheet(ims, caps, os.path.join(OUT, f"{blk}_{ch}_routes.png"), cols=6, size=150, boxes=boxes))
np.savez(os.path.join(OUT, f"{blk}_{ch}_routes.npz"), U=U, Gu=Gu, picks=picks, pi=pi, null=null)
