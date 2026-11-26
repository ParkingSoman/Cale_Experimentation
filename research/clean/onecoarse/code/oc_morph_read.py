# scale: neurons=s3:69, A:children; net=A; photos=saved; where=mac
"""Read the parent's write, child by child (LOG "The parent becomes the coarse neuron", step 7). Mac numpy, saved file.
The write map of a (k, v) pair = the stage-4 response to a unit impulse of p, at all 4 stride parities (the impulse at an
even cell only meets the centre tap of k; oc_morph.py's printed cosine used that one parity only).
Per child i: M_i = v_i applied to (k_i applied to the impulse). Calculated map = sum_i M_i.
Trained map T: least squares T ~ sum_i c_i M_i. c_i = how much of child i's path training keeps; R2 = share of T inside
the children's paths."""
import sys

import numpy as np

d = np.load(sys.argv[1] if len(sys.argv) > 1 else "results/oc_morph.npz")


def conv(x, w, stride, pad):                       # x: C x H x W, w: O x C x 3 x 3
    C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Ho, Wo = (H + 2 * pad - 3) // stride + 1, (W + 2 * pad - 3) // stride + 1
    out = np.zeros((w.shape[0], Ho, Wo))
    for a in range(3):
        for b in range(3):
            patch = xp[:, a:a + stride * Ho:stride, b:b + stride * Wo:stride]
            out += np.einsum("oc,chw->ohw", w[:, :, a, b], patch)
    return out


def paths(k, v):                                   # children x 4 parities x 512 x 7 x 7
    res = []
    for i in range(k.shape[0]):
        ms = []
        for (r, c) in [(6, 6), (6, 7), (7, 6), (7, 7)]:
            x = np.zeros((1, 14, 14)); x[0, r, c] = 1
            ms.append(conv(conv(x, k[i:i + 1], 2, 1), v[:, i:i + 1], 1, 1))
        res.append(np.stack(ms))
    return np.stack(res)


def cos(a, b):
    a, b = a.ravel(), b.ravel()
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


kids = d["order"][:11]
Mc = paths(d["k_cal"], d["v_cal"])
cal = Mc.sum(0)
A = Mc.reshape(11, -1).T
print("child share of the calculated map (norm of own path / norm of sum):",
      {int(n): round(float(np.linalg.norm(Mc[i]) / np.linalg.norm(cal)), 2) for i, n in enumerate(kids)})
T = {s: paths(d[f"tr{s}_k"], d[f"tr{s}_v"]).sum(0) for s in (0, 1)}
for s in (0, 1):
    c, *_ = np.linalg.lstsq(A, T[s].ravel(), rcond=None)
    r2 = 1 - np.sum((A @ c - T[s].ravel()) ** 2) / np.sum(T[s].ravel() ** 2)
    print(f"seed {s}: cos(cal, trained) {cos(cal, T[s]):.3f}, norm ratio {np.linalg.norm(T[s]) / np.linalg.norm(cal):.2f}, "
          f"R2 in children's paths {r2:.3f}")
    print("   c_i:", {int(n): round(float(x), 2) for n, x in zip(kids, c)})
print(f"seed 0 vs seed 1: cos {cos(T[0], T[1]):.3f}")
for s_ in (0, 1):                                  # one common scale alone (advisor check: do 11 weights beat 1?)
    c1 = (A.sum(1) @ T[s_].ravel()) / (A.sum(1) @ A.sum(1))
    r21 = 1 - np.sum((A.sum(1) * c1 - T[s_].ravel()) ** 2) / np.sum(T[s_].ravel() ** 2)
    print(f"seed {s_}: one common scale {c1:.2f}, R2 {r21:.3f}")
rng = np.random.default_rng(0)
r2n = []
for _ in range(20):
    kr = rng.standard_normal(d["k_cal"].shape); vr = rng.standard_normal(d["v_cal"].shape)
    Ar = paths(kr, vr).reshape(11, -1).T
    c, *_ = np.linalg.lstsq(Ar, T[0].ravel(), rcond=None)
    r2n.append(1 - np.sum((Ar @ c - T[0].ravel()) ** 2) / np.sum(T[0].ravel() ** 2))
print(f"R2 null (11 random paths, seed 0 map): median {np.median(r2n):.3f}, max {np.max(r2n):.3f}")
