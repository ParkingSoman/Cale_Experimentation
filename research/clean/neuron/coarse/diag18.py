"""N11 checks after step 4 (cluster). On the 19,000 held + test photos:
(a) the built coarse wheel neuron R (pre_R = sum of the 8 wheel children's pre + t, t from theory.json): per-photo peak
    input (max over positions), to correlate with B k128 n122 on the Mac;
(b) firing rates (share of positions with input > 0): B k128 n122, each wheel child of A, "any child", R;
(c) Q3, does B keep the fine split: for A and each B, accuracy among a class group's own classes on that group's photos
    (in-group accuracy, chance = 1 / group size) and the group's probability mass. Groups: 11 cars, wheel classes (29),
    face classes (47).
Saved: results/coarse/r18/diag.json, diag_R.npz.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import r18
from capture18 import net_for

torch.set_grad_enabled(False)
G1 = [79, 81, 37, 297, 496, 303, 458, 103]
th = json.load(open(os.path.join(r18.OUT, "theory.json")))
t = th["coarse"]["R_G1"]["t"]
groups = {"cars": r18.CARS, "wheel": th["groups"]["G1_classes"], "face": th["groups"]["G2_classes"]}
A = r18.load_a()
nets = {"A": A, **{tg: net_for(tg) for tg in ("B32aug_s0", "B128aug_s0", "B512aug_s0")}}
lab_all = r18.net.Photos("read").lab
caches = [r18.Cache("held"), r18.Cache("test")]

Rpos, rates = [], {"B122": [0, 0], "R": [0, 0], "any": [0, 0], **{f"A{c}": [0, 0] for c in G1}}
acc = {n: {g: [0, 0, 0.0] for g in groups} for n in nets}
for c in caches:
    for a in range(0, len(c), 250):
        rows = np.arange(a, min(a + 250, len(c)))
        x, _ = c.get(rows)
        lab = lab_all[c.ix[rows]]
        preA = A.layer4[0].bn1(A.layer4[0].conv1(x))
        preR = preA[:, G1].sum(1) + t
        Rpos.append(preR.amax((1, 2)).cpu().numpy())
        preB = nets["B128aug_s0"].layer4[0].bn1(nets["B128aug_s0"].layer4[0].conv1(x))[:, 122]
        n = preR.numel()
        for k, v in (("B122", preB), ("R", preR), ("any", (preA[:, G1] > 0).any(1).float() - .5)):
            rates[k][0] += float((v > 0).sum()); rates[k][1] += n
        for ch in G1:
            rates[f"A{ch}"][0] += float((preA[:, ch] > 0).sum()); rates[f"A{ch}"][1] += n
        for name, m in nets.items():
            p = F.softmax(r18.run4(m, x), 1).cpu()
            for g, cls in groups.items():
                mask = np.isin(lab, cls)
                if not mask.any():
                    continue
                ci = torch.as_tensor(cls)
                pg = p[torch.as_tensor(mask)][:, ci]
                yi = torch.as_tensor([list(cls).index(int(v)) for v in lab[mask]])
                acc[name][g][0] += int((pg.argmax(1) == yi).sum()); acc[name][g][1] += int(mask.sum())
                acc[name][g][2] += float(pg.sum(1).sum())
res = {"rates": {k: v[0] / v[1] for k, v in rates.items()},
       "in_group": {n: {g: dict(acc=v[0] / v[1], mass=v[2] / v[1], n=v[1], chance=1 / len(groups[g])) for g, v in d.items()}
                    for n, d in acc.items()}}
np.savez(os.path.join(r18.OUT, "diag_R.npz"), Rpos=np.concatenate(Rpos))
json.dump(res, open(os.path.join(r18.OUT, "diag.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
