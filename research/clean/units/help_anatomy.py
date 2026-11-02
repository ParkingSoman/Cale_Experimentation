# serves: TU10, TU13 (UNIT-BRANCH.md)
"""Anatomy of 'help': in the least joint free build of the gated circle toy, where does each box unit fire, what does
it write, and how much of the joint gain does each one carry? Also: how much do S's inputs vary on T's points."""
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from toy_district import make, kl, build, fit  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
X, Z, Wo = make("gated")
yN = torch.relu(Z) @ Wo.T
S, T = [0, 1, 2, 3], [4, 5, 6, 7]
onV = (Z[:, S] > 0).any(1); onA = (Z[:, T] > 0).any(1)
ang = torch.atan2(X[:, 1], X[:, 0]) * 180 / math.pi
out = {"points_V": int(onV.sum()), "points_A": int(onA.sum()), "points_neither": int((~onV & ~onA).sum())}
classes = ["V1", "V2", "A1", "A2"]
for kS, kT in ((1, 1), (1, 2)):
    D, P = fit(Z, Wo, yN, [S, T], [kS, kT], cover=False, starts=16, steps=2500)
    rec = {"joint": D, "units": []}
    for name, (Sset, m, be, V) in zip(("S(V)", "T(A)"), P):
        pre = Z[:, Sset] @ m + be
        h = torch.relu(pre)
        for j in range(h.shape[1]):
            onj = h[:, j] > 0
            # drop this unit (write 0) and see how much the joint loss rises
            P2 = [(s, mm, bb, vv.clone()) for s, mm, bb, vv in P]
            idx = 0 if name == "S(V)" else 1
            P2[idx][3][:, j] = 0.
            rise = kl(yN, build(yN, Z, Wo, P2, False)).item() - D
            rec["units"].append(dict(box=name, unit=j,
                                     on_V_points=int((onj & onV).sum()), on_A_points=int((onj & onA).sum()),
                                     on_angles=[round(float(a), 1) for a in ang[onj]][:60],
                                     write={c: round(float(w), 3) for c, w in zip(classes, V[:, j])},
                                     read_weights=[round(float(w), 3) for w in m[:, j]], bias=round(float(be[j]), 3),
                                     loss_rise_if_removed=round(rise, 4)))
    out[f"free_joint_{kS}{kT}"] = rec
    print(kS, kT, json.dumps(rec, indent=0)[:3000], flush=True)
# how much S's pre-activations vary across T's points (the information S's box could read there)
zS_on_A = Z[onA][:, S]
out["S_preact_range_on_A_points"] = [[round(float(zS_on_A[:, i].min()), 2), round(float(zS_on_A[:, i].max()), 2)] for i in range(4)]
json.dump(out, open(os.path.join(os.path.dirname(__file__), "help_anatomy.json"), "w"), indent=1)
print(out["S_preact_range_on_A_points"], out["points_V"], out["points_A"], out["points_neither"])
