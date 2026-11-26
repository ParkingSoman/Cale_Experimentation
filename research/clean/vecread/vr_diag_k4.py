"""Diagnose K4 logistic misses (description only): for each probe, its AUC for c on held half B, the rank of c in the
reading, the top entry, and the top entry's whitened cosine with c's direction.
Usage: python vr_diag_k4.py s4b1"""
import sys

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

import vecread as vr
from vr_check import gather

space = sys.argv[1]
rng = np.random.default_rng(0)
dic = vr.Dictionary(space)
held = vr.Held()
C, G = vr.SPACES[space]
src = held.src
half_a = np.zeros(held.n_img, bool)
half_a[::2] = True
ia = np.nonzero(half_a)[0]
XA = gather(held, space, rng.choice(ia, 40000), rng.integers(0, G * G, 40000)).astype(np.float64)
mu, sdv = XA.mean(0), XA.std(0) + 1e-6
KK = np.nonzero(dic.kept)[0]
Dw = dic.unit(True, only_kept=False)
rows = []
for k in rng.choice(KK, 80, replace=False):
    p, cc = held.cells(dic, k)
    res = []
    for half in (half_a, ~half_a):
        a = half[p]
        sel = rng.choice(np.nonzero(a)[0], size=min(a.sum(), 3000), replace=False) if a.sum() >= 20 else None
        if sel is None:
            break
        xp = gather(held, space, p[sel], cc[sel]).astype(np.float64)
        pool = np.nonzero(half & np.isin(src, np.unique(src[p])))[0]
        ni, nc = rng.choice(pool, 6000), rng.integers(0, G * G, 6000)
        mine = set(zip(p.tolist(), cc.tolist()))
        keep = np.array([(i, c) not in mine for i, c in zip(ni.tolist(), nc.tolist())])
        xn = gather(held, space, ni[keep][:3000], nc[keep][:3000]).astype(np.float64)
        res.append((xp, xn))
    if len(res) < 2:
        continue
    (xp, xn), (bp, bn) = res
    Z = (np.concatenate([xp, xn]) - mu) / sdv
    y = np.r_[np.ones(len(xp)), np.zeros(len(xn))]
    clf = LogisticRegressionCV(Cs=[1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1], cv=3, scoring="roc_auc", max_iter=300).fit(Z, y)
    w = clf.coef_[0] / sdv
    auc_a = roc_auc_score(y, Z @ clf.coef_[0])
    auc_b = roc_auc_score(np.r_[np.ones(len(bp)), np.zeros(len(bn))], np.r_[bp, bn] @ w)
    g = vr.geometry(dic, dic.pattern(w), whiten=True, top=len(KK))
    names = [x[0] for x in g["up"]]
    rank = names.index(dic.names[k]) if dic.names[k] in names else -1
    top = g["_top"]
    rows.append((rank, auc_a, auc_b, dic.names[k], dic.cat[k], dic.names[top], dic.cat[top], float(Dw[k] @ Dw[top]), int(dic.n[k])))
rows.sort(key=lambda r: r[0])
hit = np.array([r[0] == 0 for r in rows])
aucb = np.array([r[2] for r in rows])
print(f"hits {hit.mean():.2f} of {len(rows)}; AUC on half B: hits median {np.median(aucb[hit]):.3f}, misses median {np.median(aucb[~hit]):.3f}; train AUC median {np.median([r[1] for r in rows]):.3f}")
print("misses: rank of c | AUC train/half B | c (cat) -> top (cat) | whitened cos(c, top) | build cells of c")
for r in rows:
    if r[0] != 0:
        print(f"  {r[0]:3d} | {r[1]:.2f}/{r[2]:.2f} | {r[3]} ({r[4]}) -> {r[5]} ({r[6]}) | {r[7]:.2f} | {r[8]}")
from collections import Counter
print("top entries of misses:", Counter(r[5] for r in rows if r[0] != 0).most_common(6))
print("miss whitened cos(c, top) median:", np.median([r[7] for r in rows if r[0] != 0]))
