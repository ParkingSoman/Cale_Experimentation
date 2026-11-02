# serves: TU5, TU3, TU7 (UNIT-BRANCH.md)
"""Checks on random ReLU networks.
TU5: two unlinked branches across two hidden layers, with disjoint supports (built by a sign gate on x1); random
covering boxes (layer-1 folds m >= 0, tau >= 0; layer-2 units read the layer-1 box units with m >= 0, tau >= 0).
Check, at every x, that the joint build's logits equal the right single build's logits, and that D adds.
TU3: last-layer sets that fire together; random free boxes; check the bound (1/2) E|dS||dT| on the KL interaction.
TU7: the instance u = relu(x), v = relu(x - 1), P = {1/2, 2}, D2: exact interaction -4 at budgets (0, 2).
"""
import json
import os
from fractions import Fraction as Fr

import numpy as np

rng = np.random.default_rng(1)
relu = lambda t: np.maximum(t, 0)


def kl(yN, yB):
    def ls(y):
        m = y.max(1, keepdims=True)
        return y - m - np.log(np.exp(y - m).sum(1, keepdims=True))
    a, b = ls(yN), ls(yB)
    return (np.exp(a) * (a - b)).sum(1)


out = {"TU5": [], "TU3": []}
# ---------- TU5 ----------
for trial in range(50):
    d, n1, n2, ny, npts = 4, 5, 4, 3, 60
    X = rng.normal(size=(npts, d)); X[:, 0] = np.sign(X[:, 0]) * (1 + np.abs(X[:, 0]))
    p = rng.random(npts); p /= p.sum()
    M = 6.0
    # branch s = +1 and -1: layer-1 units W x + s*M*x1 - (M + bias), off when s*x1 < 0
    def branch(s):
        W1 = rng.normal(size=(n1, d)); W1[:, 0] = 0; b1 = rng.normal(size=n1) - M
        W1[:, 0] = s * M
        W2 = np.abs(rng.normal(size=(n2, n1))) * rng.choice([1, -1], size=(n2, n1), p=[.7, .3]); b2 = -np.abs(rng.normal(size=n2))
        Wo = rng.normal(size=(ny, n2))
        return W1, b1, W2, b2, Wo
    Bp, Bm = branch(1), branch(-1)
    co = rng.normal(size=ny)

    def run(br, X, box=None):
        W1, b1, W2, b2, Wo = br
        z1 = X @ W1.T + b1; h1 = relu(z1); z2 = h1 @ W2.T + b2; h2 = relu(z2)
        on = (z1 > 0).any(1) | (z2 > 0).any(1)
        if box is None:
            return h2 @ Wo.T, on
        # covering box: layer-1 units relu(z1 @ m1 - t1) with m1 >= 0, t1 >= 0 (reads members' pre-activations);
        # layer-2 units relu(u1 @ m2 - t2), m2 >= 0, t2 >= 0; logits read layer-2 box units with weights V
        m1, t1, m2, t2, V = box
        u1 = relu(z1 @ m1 - t1); u2 = relu(u1 @ m2 - t2)
        return u2 @ V.T, on

    def rbox(k1, k2):
        return (np.abs(rng.normal(size=(n1, k1))), np.abs(rng.normal(size=k1)) * .3, np.abs(rng.normal(size=(k1, k2))),
                np.abs(rng.normal(size=k2)) * .3, rng.normal(size=(ny, k2)) * 2)
    yP, onP = run(Bp, X); yM, onM = run(Bm, X)
    yN = yP + yM + co
    cS, cT = rbox(2, 1), rbox(1, 2)
    ySb, _ = run(Bp, X, cS); yTb, _ = run(Bm, X, cT)
    y_S = ySb + yM + co; y_T = yP + yTb + co; y_ST = ySb + yTb + co
    disjoint = not (onP & onM).any()
    pick = np.where(onP[:, None], y_S, np.where(onM[:, None], y_T, yN))
    ident = np.abs(y_ST - pick).max()
    DS, DT, DST = (p * kl(yN, y_S)).sum(), (p * kl(yN, y_T)).sum(), (p * kl(yN, y_ST)).sum()
    out["TU5"].append(dict(disjoint=bool(disjoint), max_id_err=float(ident), add_err=float(DST - DS - DT),
                           supp_sizes=[int(onP.sum()), int(onM.sum())]))
# ---------- TU3 ----------
for trial in range(200):
    d, n, ny, npts = 3, 8, 4, 40
    X = rng.normal(size=(npts, d)); p = rng.random(npts); p /= p.sum()
    W = rng.normal(size=(n, d)); b = rng.normal(size=n) * .5; Wo = rng.normal(size=(ny, n)); co = rng.normal(size=ny)
    Z = X @ W.T + b; H = relu(Z); yN = H @ Wo.T + co
    S, T = [0, 1, 2], [3, 4, 5]
    def freebox(G, k):
        m = rng.normal(size=(len(G), k)); be = rng.normal(size=k); V = rng.normal(size=(ny, k))
        return relu(Z[:, G] @ m + be) @ V.T - H[:, G] @ Wo[:, G].T
    dS, dT = freebox(S, 2), freebox(T, 2)
    I = (p * (kl(yN, yN + dS + dT) - kl(yN, yN + dS) - kl(yN, yN + dT))).sum()
    bound = .5 * (p * np.linalg.norm(dS, axis=1) * np.linalg.norm(dT, axis=1)).sum()
    out["TU3"].append(dict(I=float(I), bound=float(bound), ok=bool(abs(I) <= bound + 1e-12)))
# ---------- TU7 instance, exact ----------
P = [Fr(1, 2), Fr(2)]
rl = lambda t: t if t > 0 else Fr(0)
gN = [rl(x) + rl(x - 1) for x in P]
D_S0 = sum((g - rl(x - 1)) ** 2 for g, x in zip(gN, P))            # remove u, v kept
# joint: u removed, v's box = copy of v + spare covering unit relu(z_v) with write w; best w
# error at 1/2: 1/2 (spare off); at 2: 2 - w*1
w = Fr(2)
D_joint = (gN[0] - rl(P[0] - 1)) ** 2 + (gN[1] - rl(P[1] - 1) - w * rl(P[1] - 1)) ** 2
out["TU7"] = dict(D_S0=str(D_S0), D_T_copy="0", D_joint=str(D_joint), I=str(D_joint - D_S0))
s5 = out["TU5"]
out["summary"] = dict(TU5_disjoint_cases=sum(r["disjoint"] for r in s5),
                      TU5_max_id_err=max(r["max_id_err"] for r in s5 if r["disjoint"]),
                      TU5_max_add_err=max(abs(r["add_err"]) for r in s5 if r["disjoint"]),
                      TU3_all_ok=all(r["ok"] for r in out["TU3"]),
                      TU3_max_ratio=max(abs(r["I"]) / r["bound"] for r in out["TU3"] if r["bound"] > 0))
print(json.dumps(out["summary"], indent=1), json.dumps(out["TU7"]))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "u_tu5.json"), "w"), indent=1)
