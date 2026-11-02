"""Move 8 of the exploration: downstream. What does the unit DO? Remove it (set its output to 0 at every position) and
see which logits change, on the two groups of move 7 (animal faces, vehicles). Lab work.

usage: s6_downstream.py   (fixed to 4e:55)
Prediction (written before the run): on faces, removing 4e:55 lowers the logits of pointed-face classes (foxes,
cats, spitz dogs) and raises flat-faced dog classes (its negative pole's classes gain when its vote is gone); on
vehicles the effect is small (the essay: the car readers get "a little cat signal through the V neuron", which barely
reaches the output), and if it is visible it lowers animal-face classes rather than car classes.
"""
import os

import numpy as np
import torch

from lib import OUT, class_names, load_images, load_model, prep, run

CH = 55
m = load_model()
X, y = load_images()
names = class_names()
lab = np.load(os.path.join(os.path.dirname(__file__), "..", "curves", "label_map.npy"))  # dataset class -> logit
inv = {int(l): c for c, l in enumerate(lab)}
up = np.load(os.path.join(OUT, "4e_55_upstream.npz"))


def zero(h):
    h = h.clone()
    h[:, CH] = 0
    return h


def effect(img_ix):
    d, p0, p1 = [], [], []
    with torch.no_grad():
        for s in range(0, len(img_ix), 50):
            x = prep(X[img_ix[s:s + 50]])
            a = run(m, x)["logits"]
            b = run(m, x, edit={"4e": zero})["logits"]
            la, lb = torch.log_softmax(a, 1), torch.log_softmax(b, 1)
            d.append((lb - la).cpu().numpy())
            p0.append(torch.softmax(a, 1).cpu().numpy())
            p1.append(torch.softmax(b, 1).cpu().numpy())
    d, p0, p1 = np.concatenate(d), np.concatenate(p0), np.concatenate(p1)
    kl = (p0 * (np.log(p0 + 1e-12) - np.log(p1 + 1e-12))).sum(1)
    true_drop = np.array([d[k, lab[y[i]]] for k, i in enumerate(img_ix)])
    live = p0.mean(0) > 1e-3  # classes with real probability on these images (log-prob changes of near-zero classes are noise)
    return np.where(live, d.mean(0), np.nan), kl, true_drop


for nm, ix in [("faces", up["faces"]), ("vehicles", up["veh"])]:
    md, kl, td = effect(ix)
    ok = np.where(~np.isnan(md))[0]
    lo, hi = ok[np.argsort(md[ok])[:8]], ok[np.argsort(-md[ok])[:8]]
    print(f"  classes with mean probability > 1e-3: {len(ok)}")
    lname = lambda l: names[inv[int(l)]].split(",")[0] if int(l) in inv else f"logit {l}"
    print(f"{nm}: mean KL from removing 4e:55 {kl.mean():.4f}; mean change of the true class log-prob {td.mean():+.3f}")
    print("  most lowered:", [(lname(l), round(float(md[l]), 3)) for l in lo])
    print("  most raised: ", [(lname(l), round(float(md[l]), 3)) for l in hi])
