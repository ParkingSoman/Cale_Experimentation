# serves: H-map
"""Part C entry for Hypothesis H-map (COARSE-THEORY.md A.7), second map: a developer mix of dog photos and car photos,
regions "dog photos" and "car photos", at mixed4e and mixed5b of InceptionV1. Errors in the normalised form of DM2
(per unit: share of its activity outside the region that holds most of it). Judgment: held-out least-squares fit of the
constant 1 on dog photos from channels with mean share < 0.05 (channel activity = sum over positions).
Usage: python pc_map2.py [n_per_region]   (smoke: 8; full: 80)"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from inception import load_model, load_images, split, prep, to_3a, mixed

HERE = os.path.dirname(os.path.abspath(__file__))
n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 80
rng = np.random.default_rng(29)
CARS = [407, 436, 468, 511, 555, 569, 609, 627, 656, 661, 675, 717, 734, 751, 817, 864]


def forward(m, x):
    h = to_3a(m, x)
    h = mixed(m, "mixed3b", h)
    h = m.maxpool4(F.pad(h, (0, 1, 0, 1), value=float('-inf')), kernel_size=(3, 3), stride=(2, 2), padding=0,
                   ceil_mode=False)
    for n in ["mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed4e"]:
        h = mixed(m, n, h)
    h4e = h
    h = m.maxpool10(F.pad(h, (0, 1, 0, 1), value=float('-inf')), kernel_size=(3, 3), stride=(2, 2), padding=0,
                    ceil_mode=False)
    for n in ["mixed5a", "mixed5b"]:
        h = mixed(m, n, h)
    logits = m.softmax2_pre_activation_matmul(F.avg_pool2d(h, 7).reshape(-1, 1024))
    return h4e, h, logits


m = load_model()
Xall, y = load_images()
_, dev, _ = split(y)
lab = np.load(os.path.join(HERE, "label_map.npy"))
dog = dev[(y[dev] >= 151) & (y[dev] <= 268)]
car = dev[np.isin(y[dev], CARS)]
ix = np.r_[rng.choice(dog, n_per, replace=False), rng.choice(car, min(n_per, len(car)), replace=False)]
isdog = np.r_[np.ones(n_per, bool), np.zeros(min(n_per, len(car)), bool)]
H4, H5, top = [], [], []
with torch.no_grad():
    for b in range(0, len(ix), 32):
        sel = ix[b:b + 32]
        order = np.argsort(sel)
        inv = np.argsort(order)
        h4, h5, L = forward(m, prep(Xall[np.sort(sel)]))
        H4.append(h4.cpu().numpy()[inv]); H5.append(h5.cpu().numpy()[inv])
        top.append(L.cpu().numpy()[:, lab].argmax(1)[inv])
top = np.concatenate(top)
acc = float(np.mean(top == y[ix]))
w = np.where(isdog, 0.5 / isdog.sum(), 0.5 / (~isdog).sum())    # weight 1/2 per region


def layer_stats(H, name):
    n, C = H.shape[:2]
    U = H.reshape(n, C, -1)                               # units = (channel, position)
    md = np.einsum("n,ncp->cp", w * isdog, U)
    mc = np.einsum("n,ncp->cp", w * ~isdog, U)
    tot = md + mc
    live = tot > 0
    share = np.where(live, np.minimum(md, mc) / np.where(live, tot, 1), np.nan)
    ch_mean = np.nanmean(np.where(live, share, np.nan), 1)        # mean share over a channel's live units
    ch_live = live.any(1)
    chm = ch_mean[ch_live]
    act = U.sum(2)                                                 # channel activity on each image
    chd = np.einsum("n,nc->c", w * isdog, act); chc = np.einsum("n,nc->c", w * ~isdog, act)
    dogch = ch_live & (chd >= chc)
    out = dict(layer=name, n_units=int(U.shape[1] * U.shape[2]), live_units=int(live.sum()),
               zero_share_units=float(np.mean(U == 0)),
               unit_share_quantiles={q: float(np.nanquantile(share[live], q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
               unit_share_mean=float(np.nanmean(share[live])),
               channel_mean_share_quantiles={q: float(np.quantile(chm, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
               n_channels_below={t: int(np.sum(chm < t)) for t in (0.01, 0.05, 0.1, 0.2)},
               n_live_channels=int(ch_live.sum()), n_dog_side_channels=int(dogch.sum()))
    K = np.where(dogch & (np.nan_to_num(ch_mean, nan=1) < 0.05))[0]
    rows = np.where(isdog)[0]
    half = len(rows) // 2
    if len(K) > 0 and half >= 2:
        A_tr, A_te = act[rows[:half]][:, K], act[rows[half:]][:, K]
        lam = 1e-3 * np.trace(A_tr.T @ A_tr) / len(K)
        a = np.linalg.solve(A_tr.T @ A_tr + lam * np.eye(len(K)), A_tr.T @ np.ones(half))
        out["judgment"] = dict(n_K=int(len(K)), eps_judg_heldout=float(np.mean((1 - A_te @ a) ** 2)),
                               fit_on_car_photos_mean=float(np.mean(act[~isdog][:, K] @ a)))
    else:
        out["judgment"] = dict(n_K=int(len(K)))
    out["most_dog_only"] = [int(c) for c in np.argsort(np.where(dogch, np.nan_to_num(ch_mean, nan=9), 9))[:8]]
    out["most_car_only"] = [int(c) for c in np.argsort(np.where(ch_live & ~dogch, np.nan_to_num(ch_mean, nan=9), 9))[:8]]
    return out


res = dict(n_dog=int(isdog.sum()), n_car=int((~isdog).sum()), top1_acc=acc,
           mixed4e=layer_stats(np.concatenate(H4), "mixed4e"), mixed5b=layer_stats(np.concatenate(H5), "mixed5b"))
print(json.dumps(res, indent=1))
if n_per >= 64:
    json.dump(res, open(os.path.join(HERE, "results_pc_map2.json"), "w"), indent=1)
