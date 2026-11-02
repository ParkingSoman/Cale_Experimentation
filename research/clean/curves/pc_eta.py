# serves: H-active
"""Part C, H-active: the premise of Proposition TQ7 on the ResNet-18 proxy (estimator fixed before the run).
Same 100 areas. H_w = diag(sbar_w) - sbar_w sbar_w^T (the area's mean softmax), Hbar = sum of m_w H_w. On the image
of P_perp, the eigenvalues of Hbar^-1/2 H_w Hbar^-1/2 over all areas span [lo, hi]; the best eta for one common
metric c Hbar is (hi - lo)/(hi + lo). Also reported: the same with every H_w first divided by its own mass m_w
(a per-area scale, which TQ7 does not allow, to see whether scale or shape spreads the metrics)."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
Y = np.load(os.path.join(HERE, "cache/r18/y_train.npy")).astype(np.float64)
LAB = np.load(os.path.join(HERE, "cache/r18/active_labels.npz"))
n, C = Y.shape
S = np.exp(Y - Y.max(1, keepdims=True)); S /= S.sum(1, keepdims=True)
top = S.argmax(1); lab = LAB["G100"]
Hs, ms = [], []
for g in range(lab.max() + 1):
    K = np.where(lab == g)[0]; idx = np.where(np.isin(top, K))[0]
    if len(idx) < 2:
        continue
    sb = S[idx].mean(0); Hs.append(np.diag(sb) - np.outer(sb, sb)); ms.append(len(idx) / n)
Hbar = sum(m * H for m, H in zip(ms, Hs))
B = np.linalg.qr(np.eye(C)[:, :-1] - np.ones((C, C - 1)) / C)[0]      # orthonormal basis of the image of P_perp
ev, U = np.linalg.eigh(B.T @ Hbar @ B)
Wi = (U / np.sqrt(ev)) @ U.T
out = {}
for name, scale in [("one_metric", [1.0] * len(Hs)), ("per_area_mass_scaled", [1 / m for m in ms])]:
    lo, hi = np.inf, 0.0
    for H, c in zip(Hs, scale):
        e = np.linalg.eigvalsh(Wi @ (B.T @ (c * H) @ B) @ Wi)
        lo, hi = min(lo, e.min()), max(hi, e.max())
    out[name] = dict(lo=float(lo), hi=float(hi), eta=float((hi - lo) / (hi + lo)))
print(out)
json.dump(out, open(os.path.join(HERE, "results_pc_eta.json"), "w"), indent=1)
