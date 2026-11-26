"""geometry, cached checks.
(a) Non-agreeing Small neurons (fire-best Base != write-best Base): is s's mapped read a nonnegative mix of the two?
    Compare: fire-best alone, write-best alone, NNLS of both, NNLS of fire-best + a random Base row (control).
(b) How many distinct Base neurons the agreeing set claims.
(c) COV_g with g < 0 (more decorrelating than p's own read).
Metric: u_B fit covariance (cos = correlation of linear pre-activations).
"""
import json
import os

import numpy as np
import torch
from scipy.optimize import nnls

from cache import OUT

T = torch.float64
GO = os.path.join(OUT, "geometry")


def main():
    d = np.load(os.path.join(OUT, "stream_cov.npz"))
    C = torch.tensor(d["Cuu"], dtype=T) + 1e-6 * torch.eye(1024, dtype=T)
    L = torch.linalg.cholesky(C)
    W = torch.tensor(d["W1"], dtype=T)
    V = torch.tensor(np.load(os.path.join(OUT, "stream_V.npy")), dtype=T)
    Rf = np.load(os.path.join(OUT, "tools_peakR_b0.npy")).astype(np.float32)
    Rw = np.load(os.path.join(OUT, "writesR_b0.npy")).astype(np.float32)
    fb, wb = Rf.argmax(1), Rw.argmax(1)
    agree = np.where(fb == wb)[0]
    dis = np.where(fb != wb)[0]
    Wm = (W @ L).numpy()
    Wm /= np.linalg.norm(Wm, axis=1, keepdims=True)
    Vm = (V @ L).numpy()
    Vm /= np.linalg.norm(Vm, axis=1, keepdims=True)
    rng = np.random.default_rng(0)
    out = {"n_disagree": int(len(dis))}
    rows = {k: [] for k in ("fire", "write", "both", "fire_rand", "best_row")}
    for s in dis:
        v = Vm[s]
        a, b = Wm[fb[s]], Wm[wb[s]]
        r = Wm[rng.integers(4096)]
        rows["fire"].append(max(0, a @ v) ** 2)
        rows["write"].append(max(0, b @ v) ** 2)
        rows["both"].append(1 - nnls(np.stack([a, b], 1), v)[1] ** 2)
        rows["fire_rand"].append(1 - nnls(np.stack([a, r], 1), v)[1] ** 2)
        rows["best_row"].append(float((Wm @ v).max() ** 2))
    out["disagree_share_median"] = {k: float(np.median(x)) for k, x in rows.items()}
    # agreeing pairs for comparison
    sh = [(Wm[fb[s]] @ Vm[s]) ** 2 for s in agree]
    out["agree_share_p_median"] = float(np.median(sh))
    out["agree_distinct_base"] = int(len(set(fb[agree].tolist())))
    out["all_fire_best_distinct_base"] = int(len(set(fb.tolist())))
    # COV_g, g<0
    ev, U = torch.linalg.eigh(C)
    P = torch.tensor(fb[agree])
    vn = torch.tensor(Vm[agree])
    for g in (-0.5, -0.25, -0.1, 0.0, 0.1):
        Q = (W[P] @ ((U * ev ** g) @ U.T)) @ L
        Q = Q / Q.norm(dim=1, keepdim=True)
        out[f"COV_{g}"] = float((Q * vn).sum(1).median())
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(GO, "twoparent.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
