# serves: H-map
"""Part C check of the measure behind H-map's second map (COARSE-THEORY.md Part C): can the share measure see a known
specific unit? Same images as pc_map2.py (80 dog, 80 car dev photos, seed 29). Units: mixed4c:447 (Zoom In's car
detector) and the 10 most car-side and 10 most dog-side channels of mixed5b by logit write. Shares of the wrong region:
(a) activity summed over positions (the old measure), (b) spatial maximum per image, (c) strong firing only (images where
the spatial maximum is above the channel's 90th percentile over P)."""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from inception import load_model, load_images, split, prep, to_3a, mixed

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(29)
n_per = 80
CARS = [407, 436, 468, 511, 555, 569, 609, 627, 656, 661, 675, 717, 734, 751, 817, 864]
DOGS = list(range(151, 269))


def forward(m, x):
    h = to_3a(m, x)
    h = mixed(m, "mixed3b", h)
    h = m.maxpool4(F.pad(h, (0, 1, 0, 1), value=float('-inf')), kernel_size=(3, 3), stride=(2, 2), padding=0,
                   ceil_mode=False)
    out = {}
    for n in ["mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed4e"]:
        h = mixed(m, n, h)
        out[n] = h
    h = m.maxpool10(F.pad(h, (0, 1, 0, 1), value=float('-inf')), kernel_size=(3, 3), stride=(2, 2), padding=0,
                    ceil_mode=False)
    for n in ["mixed5a", "mixed5b"]:
        h = mixed(m, n, h)
    out["mixed5b"] = h
    return out


m = load_model()
Xall, y = load_images()
_, dev, _ = split(y)
lab = np.load(os.path.join(HERE, "label_map.npy"))
dog = dev[(y[dev] >= 151) & (y[dev] <= 268)]
car = dev[np.isin(y[dev], CARS)]
ix = np.r_[rng.choice(dog, n_per, replace=False), rng.choice(car, n_per, replace=False)]   # same draw as pc_map2.py
isdog = np.r_[np.ones(n_per, bool), np.zeros(n_per, bool)]
acts = {"mixed4c": [], "mixed5b": []}
with torch.no_grad():
    for b in range(0, len(ix), 32):
        sel = ix[b:b + 32]
        inv = np.argsort(np.argsort(sel))
        o = forward(m, prep(Xall[np.sort(sel)]))
        for k in acts:
            acts[k].append(o[k].cpu().numpy()[inv])
acts = {k: np.concatenate(v) for k, v in acts.items()}


def shares(H, c):
    a = H[:, c].reshape(len(H), -1)
    s_sum, s_max = a.sum(1), a.max(1)
    out = {}
    for name, v in (("a_sum", s_sum), ("b_max", s_max)):
        md, mc = v[isdog].mean(), v[~isdog].mean()
        out[name] = float(min(md, mc) / (md + mc)) if md + mc > 0 else None
        out[name + "_side"] = "dog" if md >= mc else "car"
    thr = np.quantile(s_max, 0.9)
    strong = s_max > thr
    if strong.sum() > 0:
        fd = strong[isdog].mean(); fc = strong[~isdog].mean()
        out["c_strong"] = float(min(fd, fc) / (fd + fc)) if fd + fc > 0 else None
        out["c_strong_side"] = "dog" if fd >= fc else "car"
        out["n_strong"] = int(strong.sum())
    return out


Wout = m.softmax2_pre_activation_matmul.weight.detach().cpu().numpy()     # (1008, 1024)
write = Wout[lab[CARS]].mean(0) - Wout[lab[DOGS]].mean(0)
car_side = np.argsort(-write)[:10]
dog_side = np.argsort(write)[:10]
res = dict(car_detector_4c_447=shares(acts["mixed4c"], 447),
           mixed5b_car_side=[dict(ch=int(c), **shares(acts["mixed5b"], c)) for c in car_side],
           mixed5b_dog_side=[dict(ch=int(c), **shares(acts["mixed5b"], c)) for c in dog_side])
# layer medians of each measure, for comparison
for L in ("mixed4c", "mixed5b"):
    allc = [shares(acts[L], c) for c in range(acts[L].shape[1])]
    res[L + "_median"] = {k: float(np.nanmedian([d.get(k) if d.get(k) is not None else np.nan for d in allc]))
                          for k in ("a_sum", "b_max", "c_strong")}
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(HERE, "results_pc_map3.json"), "w"), indent=1)
