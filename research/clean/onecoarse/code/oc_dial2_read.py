# scale: neurons=s3:69, A:children; net=A; photos=saved; where=mac
"""Read oc_dial2.npz (LOG "Step 2: the weak range itself"). Mac numpy, saved data only.
F1: peak < 2.92: mean KL(delete_weak) > .5 x mean KL(delete).
F2: photos whose top-1 delete_weak changes: share with base margin below the median DEV margin > .75.
G1: recovery(child_relu) - recovery(morph): >= +.15 in the weak range (peak .5-2), < +.05 above 2.92.
S1: weak range: mean KL(delete_otherroute) > .5 x mean KL(delete).
Also every build by peak bin (KL ratio to delete), and top-1 change rates.
"""
import sys

import numpy as np

d = np.load(sys.argv[1] if len(sys.argv) > 1 else "results/oc_dial2.npz")
pk, mg, t1 = d["dev_peak"], d["dev_margin"], d["dev_top1"]
M = ["delete", "morph", "delete_weak", "delete_strong", "delete_otherroute", "delete_parentroute", "child_relu"]
kl = {m: d[f"s2_{m}_kl"] for m in M}
top = {m: d[f"s2_{m}_top1"] for m in M}
BINS = [(0, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 2.92), (2.92, 99)]
weak, below, strong = (pk >= .5) & (pk < 2), pk < 2.92, pk >= 2.92
rng = np.random.default_rng(0)
print("own max of the 11:", np.round(d["own_max"], 2))
print("\nmean KL / mean KL(delete), by parent-peak bin (recovery = 1 - this for morph and child_relu)")
print("bin        n    " + " ".join(f"{m[:14]:>14s}" for m in M[1:]))
for lo, hi in BINS:
    m = (pk >= lo) & (pk < hi)
    print(f"{lo:4.2f}-{hi:5.2f} {m.sum():4d} " + " ".join(f"{kl[k][m].mean() / kl['delete'][m].mean():14.2f}" for k in M[1:]))
print("\ntop-1 changed (share of DEV photos):", {k: round(float(np.mean(top[k] != t1)), 3) for k in M})


def ratio_ci(a, m, B=1000):
    ii = np.nonzero(m)[0]
    v = [kl[a][b].mean() / kl["delete"][b].mean() for b in (ii[rng.integers(0, len(ii), len(ii))] for _ in range(B))]
    return kl[a][m].mean() / kl["delete"][m].mean(), np.percentile(v, [2.5, 97.5])


r, c = ratio_ci("delete_weak", below)
print(f"\nF1 peak < 2.92: KL(delete_weak)/KL(delete) {r:.2f} [{c[0]:.2f}, {c[1]:.2f}] -> {'PASS' if r > .5 else 'FAIL'}")
r2, c2 = ratio_ci("delete_strong", below)
print(f"   (delete_strong ratio {r2:.2f} [{c2[0]:.2f}, {c2[1]:.2f}])")
ch = top["delete_weak"] != t1
share = np.mean(mg[ch] < np.median(mg)) if ch.sum() else np.nan
print(f"F2 photos changed by delete_weak: {ch.sum()}; base margin below the DEV median: {share:.2f} -> "
      f"{'PASS' if share > .75 else 'FAIL'}")
for k in ["delete", "delete_strong", "delete_otherroute", "delete_parentroute"]:
    cc = top[k] != t1
    print(f"   {k}: changed {cc.sum()}, below-median margin {np.mean(mg[cc] < np.median(mg)):.2f}")
pct = lambda v: 100 * np.mean(mg[None, :] < np.atleast_1d(v)[:, None], 1)
for k in ["delete", "delete_weak", "delete_otherroute"]:
    cc = top[k] != t1
    q = pct(mg[cc])
    print(f"   {k}: changed photos' base margin as a DEV percentile: median {np.median(q):.1f}, 90th {np.percentile(q, 90):.1f}, max {q.max():.1f}"
          f" (margin values median {np.median(mg[cc]):.2f}, max {mg[cc].max():.2f}; DEV median margin {np.median(mg):.2f})")
d1 = np.load("results/oc_dial.npz")
print(f"sanity: morph KL here vs step-1 set11_morph: max diff {np.abs(kl['morph'] - d1['set11_morph']).max():.2e}")
g_w = (1 - kl["child_relu"][weak].mean() / kl["delete"][weak].mean()) - (1 - kl["morph"][weak].mean() / kl["delete"][weak].mean())
g_s = (1 - kl["child_relu"][strong].mean() / kl["delete"][strong].mean()) - (1 - kl["morph"][strong].mean() / kl["delete"][strong].mean())
print(f"G1 child_relu - morph recovery: weak {g_w:+.2f}, above 2.92 {g_s:+.2f} (n {strong.sum()}) -> "
      f"{'PASS' if g_w >= .15 and g_s < .05 else 'FAIL'}")
r3, c3 = ratio_ci("delete_otherroute", weak)
r4, c4 = ratio_ci("delete_parentroute", weak)
print(f"S1 weak range: KL(delete_otherroute)/KL(delete) {r3:.2f} [{c3[0]:.2f}, {c3[1]:.2f}] -> {'PASS' if r3 > .5 else 'FAIL'}; "
      f"parentroute {r4:.2f} [{c4[0]:.2f}, {c4[1]:.2f}]")
