# serves: TM4
"""Exact check of Proposition TM4 part 2 (COARSE-THEORY.md B.M): a tiny district error, amplified by a kept unit read
with weight 1/delta, breaks the split. Two logits (gap model), P = {a, b, c} uniform.
Units (values at a, b, c): s (1,0,0) region {a}; j1 (1,delta,0) region {a}; t1 (0,1,0) region {b,c};
j2 (0,1,1) region {b,c}, judgment. N's gap = 1.0 s + 0.0 j1 + 2.0 t1 - 0.3 j2 (N does not read j1; the free logits of a build may). One group T = {t1} at size 0.
V_b: free logits on s, j1, j2 + bias (convex). L_T: region {b,c}, kept j2 + one offset (convex)."""
import json
import os

import numpy as np

p = np.ones(3) / 3


def sig(g):
    return 1 / (1 + np.exp(-g))


def kl2(q1, g, w):
    s1 = sig(g)
    return float(np.sum(w * (q1 * np.log(q1 / s1) + (1 - q1) * np.log((1 - q1) / (1 - s1)))))


def fit(q1, F, w):
    th = np.zeros(F.shape[1])
    for _ in range(300):
        s1 = sig(F @ th)
        g = F.T @ (w * (s1 - q1))
        H = F.T @ (F * (w * s1 * (1 - s1))[:, None]) + 1e-15 * np.eye(len(th))
        step = np.linalg.lstsq(H, g, rcond=None)[0]
        th -= step
        if np.abs(step).max() < 1e-14:
            break
    return kl2(q1, F @ th, w)


out = []
for delta in (0.1, 0.01, 0.001):
    s = np.array([1, 0, 0.]); j1 = np.array([1, delta, 0]); t1 = np.array([0, 1, 0.]); j2 = np.array([0, 1, 1.])
    gapN = 1.0 * s + 0.0 * j1 + 2.0 * t1 - 0.3 * j2
    q1 = sig(gapN)
    one = np.ones(3)
    V = fit(q1, np.c_[s, j1, j2, one], p)
    LT = fit(q1, np.c_[j2, one], p * np.array([0, 1, 1.]))
    eps_dist_j1 = p[1] * delta / (p[0] * 1 + p[1] * delta)
    out.append(dict(delta=delta, V_b=V, L_T=LT, gap=abs(V - LT), eps_dist_j1=eps_dist_j1))
    print(out[-1])
ok = all(o["V_b"] < 1e-9 for o in out) and max(o["L_T"] for o in out) - min(o["L_T"] for o in out) < 1e-6 \
    and min(o["L_T"] for o in out) > 1e-4
print("PREDICTION (V_b = 0, L_T fixed > 0, eps -> 0):", "HELD" if ok else "FAILED")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tm4_dist.json"), "w"), indent=1)
