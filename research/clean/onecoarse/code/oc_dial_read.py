# scale: neurons=s3:69, A:children; net=A; photos=saved; where=mac
"""Read oc_dial.npz (LOG "The dial: are the children right?"). Mac numpy, saved data only.

Per child i (top 30 R1 children): recovery_i = 1 - mean KL(morph_i) / mean KL(delete_i), over DEV photos in a
parent-peak bin. Weak range = parent peak in [0.5, 2.0). Bootstrap 1,000 over photos.
K1: top 11, weak range: count below .2 and above .5 (pass: >= 3 each).
K2: Spearman of weak-range recovery with (a) channel-69 share of read energy and (b) R1 score, over all 30
    (primary) and over the top 11 (reported). Pass: (a) > (b) on the 30.
K3: ranks 12-30 with weak-range recovery above the top 11's median (pass: >= 3).
Descriptive, at each child's own peak cell: on-rate, channel 69's part / input when on (median), and the 3 stage-3
channels with the largest mean part when on (weak-range photos).
"""
import sys

import numpy as np

d = np.load(sys.argv[1] if len(sys.argv) > 1 else "results/oc_dial.npz")
w = np.load("results/oc_children_w.npz")
pool, pk = d["pool"], d["dev_peak"]
BINS = [(0, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 2.92), (2.92, 99)]
weak = (pk >= .5) & (pk < 2.0)
rng = np.random.default_rng(0)


def rec(i, m):
    return 1 - d["one_morph"][i, m].mean() / d["one_delete"][i, m].mean()


def ci(i, m, B=1000):
    ii = np.nonzero(m)[0]
    v = [1 - d["one_morph"][i, b].mean() / d["one_delete"][i, b].mean() for b in (ii[rng.integers(0, len(ii), len(ii))] for _ in range(B))]
    return np.percentile(v, [2.5, 97.5])


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


print("DEV photos per bin:", [int(((pk >= lo) & (pk < hi)).sum()) for lo, hi in BINS], "weak range", int(weak.sum()))
s11 = 1 - d["set11_morph"].mean() / d["set11_delete"].mean()
print(f"set of 11 on DEV: {s11:.3f}; on test random: {1 - d['te_set11_morph'].mean() / d['te_set11_delete'].mean():.3f}")
for lo, hi in BINS:
    m = (pk >= lo) & (pk < hi)
    print(f"  set11 DEV peak {lo}-{hi}: {1 - d['set11_morph'][m].mean() / d['set11_delete'][m].mean():.2f}")

on = d["peak_in"] > 0                                                    # photos x 30
par = d["peak_parts"][:, :, 69].astype(np.float32)
rw = np.array([rec(i, weak) for i in range(len(pool))])
print("\nrank neuron score share69 | delete KL weak | recovery per bin ... | weak [CI] | on-rate weak, parent part/input "
      "when on (weak) | top 3 other inputs when on (weak)")
for i, n in enumerate(pool):
    row = [rec(i, (pk >= lo) & (pk < hi)) for lo, hi in BINS]
    c = ci(i, weak)
    mw = weak & on[:, i]
    frac = np.median(par[mw, i] / d["peak_in"][mw, i]) if mw.sum() else np.nan
    parts = d["peak_parts"][mw, i].astype(np.float32).mean(0); parts[69] = -np.inf
    top3 = np.argsort(-parts)[:3]
    print(f"{i+1:2d} n{n:<4d} {w['score'][n]:.2f} {w['share'][n]:.3f} | {d['one_delete'][i, weak].mean():.5f} | "
          + " ".join(f"{x:5.2f}" for x in row) + f" | {rw[i]:.2f} [{c[0]:.2f},{c[1]:.2f}] | {on[weak, i].mean():.2f} {frac:.2f} | "
          + ", ".join(f"s3:{j} {parts[j]:.2f}" for j in top3))

t11 = rw[:11]
print(f"\nK1 top 11 weak range: below .2: {int((t11 < .2).sum())}, above .5: {int((t11 > .5).sum())} -> "
      f"{'PASS' if (t11 < .2).sum() >= 3 and (t11 > .5).sum() >= 3 else 'FAIL'}")
sh, sc = w["share"][pool], w["score"][pool]
a30, b30 = spearman(rw, sh), spearman(rw, sc)
a11, b11 = spearman(t11, sh[:11]), spearman(t11, sc[:11])
print(f"K2 Spearman, 30: share {a30:.2f} vs score {b30:.2f} -> {'PASS' if a30 > b30 else 'FAIL'}; "
      f"11: share {a11:.2f} vs score {b11:.2f}")
k3 = [int(pool[i]) for i in range(11, len(pool)) if rw[i] > np.median(t11)]
print(f"K3 ranks 12-30 above the top 11's median ({np.median(t11):.2f}): {len(k3)} {k3} -> {'PASS' if len(k3) >= 3 else 'FAIL'}")
