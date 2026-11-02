# serves: H-map
"""Part C entry for Hypothesis H-map (COARSE-THEORY.md A.7): the errors of the candidate map
"dog photos / other photos" on InceptionV1's last hidden layer (mixed5b, 7 x 7 x 1024 units, read by the logits through
the 7 x 7 average pool). Regions: dev images whose label is a dog class (151-268) and dev images of the other classes,
weighted 1/2 and 1/2. Measured: exact zeros; each channel's district error (share of its activity outside the region
holding most of it); the held-out error of the best judgment claim for "dog photos" from almost-dog-only channels.
Usage: python pc_map.py [n_per_region]   (smoke: 8)"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from inception import load_model, load_images, split, prep, to_3a, mixed, DEV

HERE = os.path.dirname(os.path.abspath(__file__))
n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 128
rng = np.random.default_rng(28)


def to_5b(m, x):
    h = to_3a(m, x)
    h = mixed(m, "mixed3b", h)
    h = m.maxpool4(F.pad(h, (0, 1, 0, 1), value=float('-inf')), kernel_size=(3, 3), stride=(2, 2), padding=0,
                   ceil_mode=False)
    for n in ["mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed4e"]:
        h = mixed(m, n, h)
    h = m.maxpool10(F.pad(h, (0, 1, 0, 1), value=float('-inf')), kernel_size=(3, 3), stride=(2, 2), padding=0,
                    ceil_mode=False)
    for n in ["mixed5a", "mixed5b"]:
        h = mixed(m, n, h)
    return h   # (B, 1024, 7, 7), post-ReLU


m = load_model()
Xall, y = load_images()
_, dev, _ = split(y)
dog = dev[(y[dev] >= 151) & (y[dev] <= 268)]
oth = dev[(y[dev] < 151) | (y[dev] > 268)]
ix = np.r_[rng.choice(dog, n_per, replace=False), rng.choice(oth, n_per, replace=False)]
isdog = np.r_[np.ones(n_per, bool), np.zeros(n_per, bool)]

lab = np.load(os.path.join(HERE, "label_map.npy"))
Tcache = np.load(os.path.join(HERE, "cache", "teacher_logits.npy"), mmap_mode="r")
H, top1, maxdiff = [], [], 0.0
with torch.no_grad():
    for b in range(0, len(ix), 32):
        sel = np.sort(ix[b:b + 32])
        order = np.argsort(np.argsort(ix[b:b + 32]))
        h = to_5b(m, prep(Xall[sel]))
        logits = m.softmax2_pre_activation_matmul(F.avg_pool2d(h, 7).reshape(-1, 1024))
        L = logits.cpu().numpy()
        maxdiff = max(maxdiff, float(np.abs(L - Tcache[sel].astype(np.float32)).max()))
        H.append(h.cpu().numpy()[order])
        top1.append(L[:, lab].argmax(1)[order])
H = np.concatenate(H)            # (n, 1024, 7, 7)
top1 = np.concatenate(top1)
# the 1008 outputs are matched to the 1000 labels by the stored one-to-one map (c0_teacher.py)
acc = float(np.mean(top1 == y[ix]))
act = H.reshape(len(ix), 1024, 49).sum(2)   # activity of each channel's 49 units on each image

zero_units = float(np.mean(H == 0))
zero_channels = float(np.mean(act == 0))
mass_dog = act[isdog].mean(0) / 2           # mix weight 1/2 per region, uniform inside a region
mass_oth = act[~isdog].mean(0) / 2
tot = mass_dog + mass_oth
live = tot > 0
err = np.where(live, np.minimum(mass_dog, mass_oth) / np.where(live, tot, 1), 0)
is_dogch = live & (mass_dog >= mass_oth)
total_rel_err = float(np.minimum(mass_dog, mass_oth)[live].sum() / tot[live].sum())

# judgment for "dog photos": fit the constant 1 on dog photos from almost-dog-only channels (district error < 0.05),
# least squares on half of the dog photos, error measured on the other half (relative to the target 1)
K = np.where(is_dogch & (err < 0.05))[0]
dog_rows = np.where(isdog)[0]
half = len(dog_rows) // 2
res_judg = None
if len(K) > 0 and half >= 2:
    A_tr, A_te = act[dog_rows[:half]][:, K], act[dog_rows[half:]][:, K]
    lam = 1e-3 * np.trace(A_tr.T @ A_tr) / max(len(K), 1)
    a = np.linalg.solve(A_tr.T @ A_tr + lam * np.eye(len(K)), A_tr.T @ np.ones(half))
    res_judg = dict(n_K=int(len(K)), eps_judg_train=float(np.mean((1 - A_tr @ a) ** 2)),
                    eps_judg_heldout=float(np.mean((1 - A_te @ a) ** 2)),
                    K_activity_share_outside=float(mass_oth[K].sum() / tot[K].sum()),
                    output_on_other_photos_mean=float(np.mean(act[~isdog][:, K] @ a)))

out = dict(n_per_region=n_per, top1_acc=acc, max_abs_diff_vs_cached_logits=maxdiff, zero_share_units=zero_units, zero_share_channel_activity=zero_channels,
           n_live=int(live.sum()), n_dog_channels=int(is_dogch.sum()),
           district_err_quantiles={q: float(np.quantile(err[live], q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
           n_channels_err_below={t: int(np.sum(live & (err < t))) for t in (0.01, 0.05, 0.1, 0.2)},
           map_total_relative_district_error=total_rel_err, judgment=res_judg,
           most_dog_specific=[int(c) for c in np.argsort(np.where(is_dogch, err, 9))[:10]],
           most_other_specific=[int(c) for c in np.argsort(np.where(live & ~is_dogch, err, 9))[:10]])
print(json.dumps(out, indent=1))
if n_per >= 64:
    json.dump(out, open(os.path.join(HERE, "results_pc_map.json"), "w"), indent=1)
