# serves: TA12
"""Exact instance of Proposition TA12 (UNIT-BRANCH.md B.8): with readers that re-fit bias and weights (DU9),
a unit whose column is (1, 1) (no class probability depends on it) is tied to the set it fires with.
Two logits, P = {1, 2, 3} uniform. N: u = relu(x - 1/2) with column (1, 0), v = relu(x - 3/2) with column (1, 1).
S = {u} at budget 0. All problems are convex (logits linear in the free reader weights), solved by Newton's method."""
import json
import math
import os

import numpy as np

P = np.array([1.0, 2.0, 3.0])
p = np.ones(3) / 3


def relu(t):
    return np.maximum(t, 0.0)


def softmax2(gap):
    # class probabilities of two logits with gap y1 - y2
    s1 = 1.0 / (1.0 + np.exp(-gap))
    return np.stack([s1, 1 - s1], 1)


def kl(q, gap_b):
    s = softmax2(gap_b)
    return float(np.sum(p * np.sum(q * (np.log(q) - np.log(s)), 1)))


def fit(q, feats):
    """least KL over gap = feats @ theta (feats include the constant); Newton on the convex logistic loss."""
    th = np.zeros(feats.shape[1])
    for _ in range(200):
        g = feats @ th
        s1 = 1.0 / (1.0 + np.exp(-g))
        grad = feats.T @ (p * (s1 - q[:, 0]))
        H = feats.T @ (feats * (p * s1 * (1 - s1))[:, None]) + 1e-14 * np.eye(len(th))
        step = np.linalg.solve(H, grad)
        th -= step
        if np.abs(step).max() < 1e-15:
            break
    return th, kl(q, feats @ th)


def run(v_shift):
    hu = relu(P - 0.5)
    hv = relu(P - v_shift)
    gap_N = hu * 1.0 + hv * (1.0 - 1.0)   # u writes (1, 0); v writes (1, 1): gap contribution 0
    q = softmax2(gap_N)
    one = np.ones(3)
    # D*({v}, 0): v removed, readers keep u and re-fit: exact (v's removal moves both logits equally)
    _, Dv = fit(q, np.stack([one, hu], 1))
    # D*(S, 0): u removed, v kept, readers re-fit bias and their weight on v (DU9)
    _, DS = fit(q, np.stack([one, hv], 1))
    # joint: u and v removed, only the bias is left
    thJ, DJ = fit(q, one[:, None])
    gJ = softmax2(thJ[0] * one) - q
    gamma = (p * hv) @ gJ
    return dict(v_shift=v_shift, D_v=Dv, D_S=DS, D_joint=DJ, I=DJ - DS - Dv, gamma=gamma.tolist())


out = dict(main=run(1.5), control_never_on=run(3.5))
print(json.dumps(out, indent=1))
ok_main = out["main"]["I"] > 1e-6 and abs(out["main"]["gamma"][0]) > 1e-6 and abs(out["main"]["D_v"]) < 1e-12
ok_ctrl = abs(out["control_never_on"]["I"]) < 1e-12
print("PREDICTION main (I > 0, gamma != 0, D*({v},0) = 0):", "HELD" if ok_main else "FAILED")
print("PREDICTION control (I = 0):", "HELD" if ok_ctrl else "FAILED")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ta12_sub.json"), "w"), indent=1)
