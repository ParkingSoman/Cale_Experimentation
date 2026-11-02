# serves: CU1 (UNIT-BRANCH.md)
"""What is a district's one-unit coarse version? Single build of district V at budget 1, readers re-fit their biases."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import toy_shelter as ts  # noqa: E402
from toy_bias import make_orig, fit_bias  # noqa: E402

torch.set_default_dtype(torch.float64)


def spearman(a, b):
    ra = a.argsort().argsort().double(); rb = b.argsort().argsort().double()
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))


out = {}
for name, mk in (("orig", make_orig), ("sheltered", lambda: ts.make("sheltered", False)),
                 ("q4", lambda: ts.make("q4", False)), ("pose", lambda: ts.make("pose", False))):
    X, H1, Z2, Wo, yN, w, inb = mk()
    S = list(range(4)); onS = (Z2[:, S] > 0).any(1)
    D, (P, b) = fit_bias(H1, Z2, Wo, yN, w, [(S, inb["S"])], [1])
    _, ib, m, be, V = P[0]
    u = torch.relu(H1[:, ib] @ m + be)[:, 0]
    ang = torch.atan2(X[:, 1], X[:, 0])
    v = V[:, 0]
    r = dict(loss=round(D, 5), on_frac_V=round(float((u[onS] > 0).double().mean()), 3),
             on_outside_V=int((u[~onS] > 0).sum()), spearman_angle=round(spearman(u[onS], ang[onS]), 3),
             contrast=round(float((v[1] - v[0]) / 2), 3), common=round(float((v[1] + v[0]) / 2), 3),
             write=[round(float(t), 2) for t in v], read={int(i): round(float(t), 2) for i, t in zip(ib, m[:, 0])},
             unit_bias=round(float(be[0]), 2), logit_bias=[round(float(t), 2) for t in b])
    out[name] = r
    print(name, r, flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "cu1_dial.json"), "w"), indent=1)
