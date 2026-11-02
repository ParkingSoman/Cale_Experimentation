"""Move 7 of the exploration: upstream. Which units of the layer below build the unit, separately on two groups of
inputs (animal faces, vehicles)? Contribution of a lower unit j at the unit's best position = W[ch, j] h_j. Lab work.

usage: s5_upstream.py   (fixed to 4e:55, whose 1x1 branch reads 4d directly)
Prediction (written before the run): if 4e:55 is one concept (a converging "V" front), its strongest upstream
contributors on animal faces and on vehicles overlap (at least 3 of the top 10 in common, against about 0.2 expected
by chance from 528 units); if it packs two jobs, the two top-10 lists are nearly disjoint.
"""
import os

import numpy as np
import torch

from lib import OUT, class_names, load_images, load_model, prep, run, sheet

CH = 55
m = load_model()
X, y = load_images()
names = class_names()
W = m.mixed4e_1x1_pre_relu_conv.weight[CH, :, 0, 0].detach().cpu().numpy()  # 528 weights from 4d
scan = np.load(os.path.join(OUT, "4e_55_scan.npz"))
VEH = ["sports car", "jeep", "police van", "minivan", "convertible", "cab", "racer", "beach wagon", "limousine",
       "pickup", "grille", "Model T"]
veh_ids = [min(i for i, s in enumerate(names) if s.split(",")[0].strip() == v) for v in VEH]


def contributions(img_ix):
    out, best = [], []
    with torch.no_grad():
        for s in range(0, len(img_ix), 50):
            b = img_ix[s:s + 50]
            r = run(m, prep(X[b]), want_pre=("4e",))
            z = r["pre_4e"][:, CH].reshape(len(b), -1)
            k = z.argmax(1)
            H = r["4d"].shape[-1]
            h = r["4d"][torch.arange(len(b)), :, k // H, k % H].cpu().numpy()  # B x 528 at the best position
            out.append(h * W[None, :])
            best.append(z.max(1)[0].cpu().numpy())
    return np.concatenate(out), np.concatenate(best)


faces = scan["ix"][np.argsort(-scan["mx"])][:400]
faces = faces[y[faces] < 398][:100]  # animal classes come first in the label order
veh_all = np.where(np.isin(y, veh_ids))[0]
cv_all, bv_all = contributions(veh_all)
order = np.argsort(-bv_all)[:100]
cv, veh = cv_all[order], veh_all[order]
cf, bf = contributions(faces)
mf, mv = cf.mean(0), cv.mean(0)
topf, topv = np.argsort(-mf)[:10], np.argsort(-mv)[:10]
negf, negv = np.argsort(mf)[:5], np.argsort(mv)[:5]
common = sorted(set(topf) & set(topv))
print(f"faces: 100 images, mean unit value {bf.mean():.0f}; vehicles: best 100 of {len(veh_all)}, mean {bv_all[order].mean():.0f}")
print("top-10 4d contributors on faces:   ", [(int(j), round(float(mf[j]))) for j in topf])
print("top-10 4d contributors on vehicles:", [(int(j), round(float(mv[j]))) for j in topv])
print(f"in common: {common} ({len(common)} of 10; chance about {10 * 10 / 528:.2f})")
print("most negative on faces:   ", [(int(j), round(float(mf[j]))) for j in negf])
print("most negative on vehicles:", [(int(j), round(float(mv[j]))) for j in negv])
share_f = mf[topf].sum() / mf[mf > 0].sum()
share_v = mv[topv].sum() / mv[mv > 0].sum()
print(f"share of positive contribution in the top 10: faces {share_f:.2f}, vehicles {share_v:.2f}")
np.savez(os.path.join(OUT, "4e_55_upstream.npz"), mf=mf, mv=mv, faces=faces, veh=veh)

# what are these 4d units? top images of each selected unit, from 3000 random images
sel = list(dict.fromkeys(list(topf[:5]) + list(topv[:5]) + list(negf[:2])))
rng = np.random.default_rng(2)
pool = np.sort(rng.choice(len(y), 3000, replace=False))
M, P = np.zeros((3000, 528)), np.zeros((3000, 528), int)
with torch.no_grad():
    for s in range(0, 3000, 100):
        h = run(m, prep(X[pool[s:s + 100]]))["4d"]
        f = h.reshape(h.shape[0], 528, -1)
        M[s:s + 100] = f.max(2)[0].cpu().numpy()
        P[s:s + 100] = f.argmax(2).cpu().numpy()
ims, caps, boxes = [], [], []
for j in sel:
    tag = ("F" if j in topf else "") + ("V" if j in topv else "") + ("-" if j in negf else "")
    for i in np.argsort(-M[:, j])[:6]:
        g = 14
        cy, cx = (P[i, j] // g + 0.5) * 16, (P[i, j] % g + 0.5) * 16
        ims.append(X[pool[i]]); caps.append(f"4d:{j} {tag} {names[y[pool[i]]]}")
        boxes.append((cx - 40, cy - 40, cx + 40, cy + 40))
print(sheet(ims, caps, os.path.join(OUT, "4e_55_upstream.png"), cols=6, size=150, boxes=boxes))
print("rows (4d units, F = top on faces, V = top on vehicles, - = most negative on faces):", sel)
