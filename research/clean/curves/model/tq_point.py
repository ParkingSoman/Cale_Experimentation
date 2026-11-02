# serves: TQ8
"""Check of Lemma TQ8 (COARSE-THEORY.md B.Q): g(-omega) (1/2) Var_s(d) <= KL(s || softmax(y + d)) <= g(omega) (1/2)
Var_s(d), omega = max d - min d, g(c) = 2 (e^c - 1 - c) / c^2. Random logits y (from flat to very confident) and changes
d (from tiny to large spread, including a large negative push on unlikely classes, the packing move). Also reports the
ratio KL / ((1/2) Var) for that push, to show the quadratic overstating the cost."""
import json
import os

import numpy as np
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1102)


def g(c):
    if abs(c) < 1e-6:
        return 1.0 + c / 3.0
    return 2.0 * (np.expm1(c) - c) / (c * c)


fails = []; n = 0; push = []
for trial in range(20000):
    ny = int(rng.integers(2, 30))
    y = rng.normal(size=ny) * rng.choice([0.1, 1, 3, 8])
    d = rng.normal(size=ny) * rng.choice([1e-3, 0.1, 1, 4])
    if trial % 4 == 0:                          # push the unlikely classes far down
        lo = np.argsort(y)[: max(1, ny // 2)]
        d = np.zeros(ny); d[lo] = -rng.uniform(1, 30)
    ls = y - logsumexp(y); s = np.exp(ls)
    lq = (y + d) - logsumexp(y + d)
    kl = float(np.sum(s * (ls - lq)))
    var = float(np.sum(s * d * d) - np.sum(s * d) ** 2)
    om = float(d.max() - d.min())
    lo_b, hi_b = g(-om) * var / 2, g(om) * var / 2
    tol = 1e-10 * max(1.0, abs(kl)) + 1e-14
    n += 1
    if kl < lo_b - tol or kl > hi_b + tol:
        fails.append((trial, kl, lo_b, hi_b, om))
    if trial % 4 == 0 and var > 1e-12:
        push.append(kl / (var / 2))
print("checks", n, "fails", len(fails), fails[:3])
print("packing push: KL / ((1/2) Var) median %.3g, min %.3g" % (np.median(push), np.min(push)))
json.dump(dict(checks=n, fails=fails, push_median=float(np.median(push)), push_min=float(np.min(push))),
          open(os.path.join(HERE, "tq_point.json"), "w"), indent=1)
