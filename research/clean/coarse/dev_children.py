"""Dev coarseness metric: children per unit pole (R1 rule: child = old pole whose top-1% positions sit at unit
percentile >= 0.95 (median); near-copies with |corr| >= 0.9 excluded; hub guard 20%). Dev test blocks 0-39."""
import numpy as np


def children_stats(layer_units, Xh, G, U, q=0.01, thr=0.95, near=0.9, hub=0.2):
    import mlx.core as mx
    A = np.array(layer_units(mx.array(Xh.astype(np.float32))), dtype=np.float64)       # (N, K)
    h = (Xh @ G.T) / (1 + np.exp(-(Xh @ G.T))) * (Xh @ U.T)                              # (N, n)
    P_old = np.concatenate([h, -h], 1)
    P_new = np.concatenate([A, -A], 1)
    nh = max(1, int(len(Xh) * q))
    hot = np.argpartition(-P_old, nh, axis=0)[:nh].T                                     # (2n, nh)
    R = np.empty_like(P_new, dtype=np.float32)
    for j in range(P_new.shape[1]):
        R[np.argsort(P_new[:, j]), j] = np.linspace(0, 1, len(P_new), dtype=np.float32)
    cov = np.stack([np.median(R[:, j][hot], axis=1) for j in range(P_new.shape[1])])     # (2K, 2n)
    zn = (P_new - P_new.mean(0)) / (P_new.std(0) + 1e-9)
    zo = (P_old - P_old.mean(0)) / (P_old.std(0) + 1e-9)
    corr = np.abs(zn.T @ zo / len(Xh))
    kids = (cov >= thr) & (corr < near)
    kids &= ~(kids.mean(0) > hub)[None, :]
    k = kids.sum(1)
    return {"median_children": float(np.median(k)), "share_2plus": float(np.mean(k >= 2)), "mean_children": float(k.mean())}
