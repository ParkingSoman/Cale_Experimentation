# scale: neurons=s3:69, s3:all, A:children, A:block0-all; net=A; photos=saved; where=mac
# rung: Q=advisor checks on VECMATH step 3: are the carriers the children's own read weights (selection), and a truly quiet photo group; N=saved oc_vm_rebuild.npz + w_A.npz; EST=10 s Mac; PERMIT=Shonu approved plan "New track: deletion as vector math" 2026-11-24
"""Checks after VECMATH step 3 (Mac, saved data only)."""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
v = np.load(os.path.join(HERE, "..", "results", "oc_vm_rebuild.npz"))
w = np.load(os.path.join(HERE, "..", "..", "neuron", "results", "coarse", "r18", "w_A.npz"))
g, _, _, var = w["bn1_0"]
R = w["W1_0"] * (g / np.sqrt(var + 1e-5))[:, None, None, None]          # 512 x 256 x 3 x 3
kids = v["kids"]
kept = np.setdiff1d(np.arange(512), kids)


def cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


print("C1: skip part of beta (256) vs the child's own read, centre tap / whole 3x3 summed")
for nm in ("known", "kl"):
    B = v[f"beta_{nm}"][:, 501:757]
    own_c = [cos(B[i], R[d, :, 1, 1]) for i, d in enumerate(kids)]
    own_s = [cos(B[i], R[d].sum((1, 2))) for i, d in enumerate(kids)]
    other = [cos(B[i], R[d2].sum((1, 2))) for i, d in enumerate(kids) for d2 in kids if d2 != d]
    rnd = [cos(B[i], R[c].sum((1, 2))) for i in range(11) for c in kept[::10]]
    print(f"  {nm}: own centre median {np.median(own_c):.2f} (per child {np.round(own_c, 2).tolist()})")
    print(f"        own 3x3-summed median {np.median(own_s):.2f} | sibling {np.median(other):.2f} | random kept {np.median(rnd):.2f}")

print("\nC2: block-0 carriers - does each child's top block-0 carrier read what the child reads (channel 69 zeroed)?")
Rf0 = R.copy()
Rf0[:, 69] = 0
Rf0 = Rf0.reshape(512, -1)
allc = np.array([[cos(Rf0[c], Rf0[d]) for c in kept] for d in kids])     # 11 x 501
pd = v["perd_known"][:, :501]
pcts = []
for i, d in enumerate(kids):
    top = np.argsort(-pd[i])[:5]
    cs = allc[i, top]
    pct = [(allc[i] < x).mean() for x in cs]
    pcts += pct
    print(f"  n{d}: " + " ".join(f"n{kept[t]} (cos {c:.2f}, pct {p:.2f})" for t, c, p in zip(top, cs, pct)))
print(f"  median percentile of the top-5 block-0 carriers' read similarity among all 501 kept: {np.median(pcts):.2f}")

print("\nC3: quiet photos (lowest decile of D's summed firing, cut on DEV)")
evl = v["evl"]
te_d = v["te_d"]
q10 = np.quantile(v["dev_d"], .1)
sel = np.nonzero(te_d[evl] <= q10)[0]
print(f"  DEV 10th pct {q10:.1f}; evaluated photos below it: {len(sel)}")
for b in ("V0", "V-known", "V-KL"):
    print(f"  {b:8s} mean KL {v[f'kl_{b}'][sel].mean():.5f}")
for b in ("V-known", "V-KL"):
    dlt = v[f"kl_{b}"][sel] - v["kl_V0"][sel]
    print(f"  {b}: build - V0 mean {dlt.mean():+.5f}; worse photos {(dlt > 0).mean():.2f}, their mean rise {dlt[dlt > 0].mean():.5f}")
pos = {r: i for i, r in enumerate(evl)}
ix = np.array([pos[r] for r in v["set_te_rand"]])
k0 = v["kl_V0"][ix]
edges = np.quantile(k0, [0, .25, .5, .75, .9, 1])
print("  random 2,000 binned by delete harm: V0 range | V0 | V-known | V-KL | share worse (known) | mean rise of worse (known)")
for a, b in zip(edges[:-1], edges[1:]):
    s = (k0 >= a) & (k0 <= b)
    kk, kq = v["kl_V-known"][ix][s], v["kl_V-KL"][ix][s]
    dl = kk - k0[s]
    print(f"   {a:.4f}-{b:.4f}  {k0[s].mean():.5f}  {kk.mean():.5f}  {kq.mean():.5f}  {(dl > 0).mean():.2f}  "
          f"{dl[dl > 0].mean() if (dl > 0).any() else 0:.5f}")
print("\nC4: absolute KL left (mean)")
for sn in ("te_fired", "te_quiet", "te_strong", "te_rand"):
    ii = np.array([pos[r] for r in v[f"set_{sn}"]])
    print(f"  {sn:10s}", " ".join(f"{b} {v[f'kl_{b}'][ii].mean():.5f}" for b in ("base", "V0", "V-known", "V-KL")))
