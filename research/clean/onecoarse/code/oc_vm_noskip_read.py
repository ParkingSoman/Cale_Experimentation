# scale: neurons=s3:69, A:children; net=A; photos=saved; where=mac
"""Read oc_vm_noskip.npz (Mac, numpy). Recovery per gate set with a paired bootstrap (2,000 resamples of test photos),
the difference "everywhere minus parent" with its 95% CI, and harm by bins of D's summed firing (DEV quantiles).
Usage: python3 oc_vm_noskip_read.py [path]
"""
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "../results/oc_vm_noskip.npz"
d = np.load(path)
evl = d["evl"]
pos = {r: i for i, r in enumerate(evl)}
B = ["all758", "b0p", "b0", "p"]
rng = np.random.default_rng(0)
NB = 2000


def rec(x, v):
    return 1 - x.mean() / v.mean()


print("damping:", {b: float(d[f"lam_{b}"]) for b in B})
for g in ["te_strong", "te_fired", "te_rand", "te_quiet"]:
    ix = np.array([pos[r] for r in d["set_" + g]])
    v0 = d["kl_V0"][ix]
    k = {b: d[f"kl_{b}"][ix] for b in B}
    boot = np.array([[rec(k[b][s], v0[s]) for b in B] for s in (rng.integers(0, len(ix), len(ix)) for _ in range(NB))])
    lo, hi = np.percentile(boot, [2.5, 97.5], 0)
    pt = [rec(k[b], v0) for b in B]
    print(f"\n[{g}] n {len(ix)}, delete KL {v0.mean():.5f}")
    for i, b in enumerate(B):
        print(f"  {b:7s} {pt[i]:.3f} [{lo[i]:.3f}, {hi[i]:.3f}]")
    for a_, b_ in (("b0p", "p"), ("b0", "p"), ("all758", "b0p"), ("b0p", "b0")):
        i, j = B.index(a_), B.index(b_)
        df = boot[:, i] - boot[:, j]
        print(f"  {a_} minus {b_}: {pt[i] - pt[j]:.3f} [{np.percentile(df, 2.5):.3f}, {np.percentile(df, 97.5):.3f}]")

# harm by D's summed firing, random test photos, cuts at DEV quantiles
ix = np.array([pos[r] for r in d["set_te_rand"]])
te_d = d["te_d"][d["set_te_rand"]]
cuts = np.quantile(d["dev_d"], [0, .1, .25, .5, .75, .9, 1])
cuts[0], cuts[-1] = -np.inf, np.inf
print("\nharm by D's firing (random test photos; DEV quantile bins)")
print("bin       n    delete    " + "  ".join(f"{b:>8s}" for b in B) + "   share worse than delete: " + " ".join(B))
for lo_, hi_, nm in zip(cuts[:-1], cuts[1:], ["0-10", "10-25", "25-50", "50-75", "75-90", "90-100"]):
    m = (te_d >= lo_) & (te_d < hi_)
    v0 = d["kl_V0"][ix][m]
    row = "  ".join(f"{d[f'kl_{b}'][ix][m].mean():8.5f}" for b in B)
    worse = " ".join(f"{(d[f'kl_{b}'][ix][m] > v0).mean():.2f}" for b in B)
    print(f"{nm:7s} {m.sum():5d}  {v0.mean():.5f}  {row}   {worse}")
