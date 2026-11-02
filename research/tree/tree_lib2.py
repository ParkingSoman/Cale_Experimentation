"""Helpers for Z3 (stacked merges)."""
import numpy as np


def parent_map(fine, coarse):
    """Map each fine group id to the coarse group id holding its members.
    Returns (map, violations) where violations = fine groups split across coarse groups."""
    kf = fine.max() + 1
    pm = np.zeros(kf, int)
    bad = 0
    for j in range(kf):
        c = np.unique(coarse[fine == j])
        pm[j] = c[0]
        bad += int(len(c) > 1)
    return pm, bad


def merge_parents(G2, U2, D2, cnt, pm):
    """Parent of parents. G2,U2 are already sign-aligned means over cnt leaves; weight by cnt so the
    result is the plain mean over all leaves. Down columns are summed."""
    k = pm.max() + 1
    G3 = np.zeros((k, G2.shape[1]), np.float64)
    U3 = np.zeros_like(G3)
    D3 = np.zeros((D2.shape[0], k), np.float64)
    c3 = np.zeros(k)
    w = cnt.astype(np.float64)
    np.add.at(G3, pm, G2 * w[:, None])
    np.add.at(U3, pm, U2 * w[:, None])
    np.add.at(c3, pm, w)
    for j in range(k):
        D3[:, j] = D2[:, pm == j].sum(1)
    return (G3 / c3[:, None]).astype(np.float32), (U3 / c3[:, None]).astype(np.float32), D3.astype(np.float32), c3
