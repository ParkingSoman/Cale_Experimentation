# scale: neurons=s3:69, B:n101, A:children; net=A, B128aug_s1; photos=saved; where=mac
"""Step 2 of "Back to one coarse neuron" (2026-11-23): what number does n101 send?

Linear fact: n101's mean pre-activation per photo ~ sum_c (kernel sum on c) x (c's photo mean) + bias.
  parent term = channel 69's part; rest term = all other channels' parts.
Check: parent + rest + bias against the saved mean0 of n101. Bar r >= .95.
T = the children's joint amount = sum over the 11 R1 children of (child score) x relu(peak pre-activation).
T-a corr(rest, parent). T-b corr(parent + rest, T) - corr(parent, T), against 200 shuffles (n101's non-69 kernel
sums moved to permuted non-69 channels). T-c on n101's top 1% photos (by its peak), the share with the parent term
below its median, and the channels that carry n101 there.
Inputs: results/oc_weights.npz, results/oc_children_w.npz, r18/read_s3.npz, read_A.npz, read_B128aug_s1.npz.
Output: printed, results/oc_q_number.npz.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
R18 = os.path.join(HERE, "..", "..", "neuron", "results", "coarse", "r18")
P, C, K = 69, 101, 11


def corr(a, b):
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    return float((a * b).mean())


def main():
    d = np.load(os.path.join(RES, "oc_weights.npz"))
    cw = np.load(os.path.join(RES, "oc_children_w.npz"))
    s3 = np.load(os.path.join(R18, "read_s3.npz"))
    mean = s3["mean"].astype(np.float64)
    pos69 = s3["pos"][:, P].astype(np.float64)
    rA = np.load(os.path.join(R18, "read_A.npz"))
    rB = np.load(os.path.join(R18, "read_B128aug_s1.npz"))
    assert (rA["ix"] == s3["ix"]).all() and (rB["ix"] == s3["ix"]).all(), "photo order"
    m101 = rB["mean0"][:, C].astype(np.float64)
    p101 = rB["pos0"][:, C].astype(np.float64)

    ksum = d["read_c101"][0].sum((1, 2)).astype(np.float64)
    b = float(d["off_c101"][0])
    parts = mean * ksum[None, :]
    parent = parts[:, P]
    rest = parts.sum(1) - parent
    recon = parent + rest + b
    r_rec = corr(recon, m101)
    print(f"reconstruction r = {r_rec:.4f} (bar .95); mean abs err {np.abs(recon - m101).mean():.3f}, "
          f"n101 mean0 sd {m101.std():.3f}")
    print(f"sd over photos: parent term {parent.std():.3f}, rest term {rest.std():.3f}")

    kids = cw["order"][:K]
    a = cw["score"][kids]
    T = (np.maximum(rA["pos0"][:, kids].astype(np.float64), 0) * a[None, :]).sum(1)

    ta = corr(rest, parent)
    c_par, c_full = corr(parent, T), corr(parent + rest, T)
    gain = c_full - c_par
    rng = np.random.default_rng(0)
    others = np.array([c for c in range(256) if c != P])
    sg = []
    for _ in range(200):
        ks = ksum.copy()
        ks[others] = ksum[rng.permutation(others)]
        rs = (mean[:, others] * ks[others][None, :]).sum(1)
        sg.append(corr(parent + rs, T) - c_par)
    sg = np.array(sg)
    # partial correlation of rest with T, given parent
    def resid(y, x):
        x1 = np.c_[np.ones_like(x), x]
        return y - x1 @ np.linalg.lstsq(x1, y, rcond=None)[0]
    pr = corr(resid(rest, parent), resid(T, parent))
    print(f"T-a corr(rest, parent) = {ta:+.3f}")
    print(f"T-b corr(parent, T) = {c_par:+.3f}; corr(parent + rest, T) = {c_full:+.3f}; gain {gain:+.3f}; "
          f"shuffled gain median {np.median(sg):+.3f}, 99th {np.percentile(sg, 99):+.3f}; "
          f"partial corr(rest, T | parent) {pr:+.3f}")
    print(f"     peak level (no split): corr(pos69, T) {corr(pos69, T):+.3f}; corr(n101 peak, T) {corr(p101, T):+.3f}; "
          f"corr(n101 peak, pos69) {corr(p101, pos69):+.3f}; corr(n101 mean, T) {corr(m101, T):+.3f}")

    n = len(p101)
    top = np.argsort(-p101)[: n // 100]
    low = parent[top] < np.median(parent)
    print(f"T-c top 1% photos by n101 peak: {len(top)}; parent term below median on {low.mean():.2f}; "
          f"parent term z on them: median {np.median((parent[top] - parent.mean()) / parent.std()):+.2f}")
    strong = pos69 >= 2.92
    thr = np.percentile(p101, 95)
    print(f"     parent-strong photos {strong.sum()}: n101 peak in its top 5% on {(p101[strong] >= thr).mean():.2f}; "
          f"top-1% photos that are parent-strong {strong[top].mean():.2f}")
    dev = parts - parts.mean(0)
    for name, sel in (("top 1% all", top), ("top 1%, parent low", top[low]), ("top 1%, parent high", top[~low])):
        if len(sel) == 0:
            continue
        m = dev[sel].mean(0)
        o = np.argsort(-m)[:8]
        print(f"     {name} ({len(sel)}): parent +{m[P]:.3f}; largest pushes up " +
              ", ".join(f"s3:{c} {m[c]:+.3f}" for c in o) + f"; sum of all {m.sum():+.3f}")
    np.savez(os.path.join(RES, "oc_q_number.npz"), parent=parent, rest=rest, T=T, r_rec=r_rec, gain=gain,
             shuf_gain=sg, top=top, partial=pr)


if __name__ == "__main__":
    main()
