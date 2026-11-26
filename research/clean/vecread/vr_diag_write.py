"""Why does the write geometry find nothing for w101 while its data-side examples say wheel? (description only)
For w101, wFold and the 11 children's writes (s4b0, 3 x 3 summed):
  1. top-5 geometry (raw and whitened) against top-5 examples (data side), and their overlap
  2. share of the whitened length in the lowest-variance directions (bottom half / bottom 10% of eigenvalues), against
     the same share for the kept concept directions
  3. tap sum norm against the single-tap norms (cancellation), and the centre tap read alone
"""
import numpy as np

import vecread as vr
from cards_n101 import CLEAN, R18, bn_scale, K  # noqa: F401
import os

A = np.load(os.path.join(R18, "w_A.npz"))
B = np.load(os.path.join(R18, "w_B128aug_s1.npz"))
cw = np.load(os.path.join(CLEAN, "onecoarse", "results", "oc_children_w.npz"))
kids = cw["order"][:K]
WA = A["W2_0"] * bn_scale(A["bn2_0"])[:, None, None, None]
WB = B["W2_0"] * bn_scale(B["bn2_0"])[:, None, None, None]
kern = {"w101": WB[:, 101], "wFold": WA[:, kids].sum(1)}
for k in kids:
    kern[f"A:n{k}"] = WA[:, k]

dic = vr.Dictionary("s4b0")
held = vr.Held()
ev, U = np.linalg.eigh(dic.cov)                       # ascending
def low_share(v, frac):
    c = U.T @ v
    wc = c / np.sqrt(np.maximum(ev, ev.max() * 1e-6))  # whitened coordinates
    n = int(len(ev) * frac)
    return float((wc[:n] ** 2).sum() / (wc ** 2).sum())

Dk = dic.D[dic.kept]
print("concept directions: whitened length in bottom 50% / 10% eigen-directions, median:",
      round(float(np.median([low_share(d, .5) for d in Dk])), 3), round(float(np.median([low_share(d, .1) for d in Dk])), 3))
print()
for name, Wk in kern.items():
    v = Wk.sum((1, 2))
    taps = np.linalg.norm(Wk.reshape(512, 9), axis=0)
    canc = np.linalg.norm(v) / taps.sum()               # 1 = taps all parallel, ~1/3 = random taps
    g_raw = vr.geometry(dic, v, whiten=False, top=5)
    g_w = vr.geometry(dic, v, whiten=True, top=5)
    ex = vr.examples(held, dic, held.project("s4b0", v))
    ex_n = [e[0] for e in ex]
    ov = lambda g: len(set(x[0] for x in g["up"]) & set(ex_n))
    print(f"{name:8s} low-var share {low_share(v, .5):.2f}/{low_share(v, .1):.2f} | sum/sum|tap| {canc:.2f} "
          f"(centre tap {taps[4] / taps.sum():.2f}) | overlap geom-examples raw {ov(g_raw)}/5, whitened {ov(g_w)}/5")
    print(f"   raw  {[(x[0], x[2]) for x in g_raw['up']]}")
    print(f"   whit {[(x[0], x[2]) for x in g_w['up']]}")
    print(f"   exmp {[(e[0], e[2]) for e in ex]}")
    if name == "w101":
        c = vr.geometry(dic, Wk[:, 1, 1], whiten=True, top=5)
        print(f"   centre tap alone (whitened) {[(x[0], x[2]) for x in c['up']]}")
