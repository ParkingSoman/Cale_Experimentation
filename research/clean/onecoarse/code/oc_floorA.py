"""Floor A scorer (Mac). Child of s3:69 = (dependence) s3:69 among j's top-K positive feeders on j's top-50 photos
AND (drive) s3:69's used vote into j (positive part, only where j is on) above the 90th percentile of the 512,
on s3:69's top-200 photos. Computed per half (h0 even rows, h1 odd rows) and on all photos (path_A feed0)."""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
S = np.load(os.path.join(R, "oc_step1.npz"))
F = np.load(os.path.join(R, "oc_floor.npz"))
ALL = np.load(os.path.join(HERE, "..", "..", "neuron", "results", "coarse", "r18", "path_A.npz"))["feed0"]
P = 69
ppk, c69, pre = S["ppk"].astype(np.float32), S["A_c69"].astype(np.float32), S["A_pre"].astype(np.float32)


def used_drive(rows):
    top = rows[np.argsort(-ppk[rows])[:200]]
    return (np.clip(c69[top], 0, None) * (pre[top] > 0)).mean(0)


def child_set(feed, drive, k):
    rank = (feed > feed[:, [P]]).sum(1)
    return set(np.where((rank < k) & (drive > np.quantile(drive, .90)))[0].tolist()), rank


halves = {h: np.arange(h, len(ppk), 2) for h in (0, 1)}
sets = {}
for k in (2, 3, 4):
    sets[("all", k)], _ = child_set(ALL, used_drive(np.arange(len(ppk))), k)
    for h in (0, 1):
        sets[(h, k)], _ = child_set(F[f"feed_h{h}"], used_drive(halves[h]), k)
J = lambda a, b: len(a & b) / max(1, len(a | b))
base = sets[("all", 3)]
print("child set (all photos, top 3):", sorted(base), "size", len(base))
print("A1 stability, Jaccard half0 vs half1 (top 3):", round(J(sets[(0, 3)], sets[(1, 3)]), 2),
      "| half0", sorted(sets[(0, 3)]), "| half1", sorted(sets[(1, 3)]))
for k in (2, 4):
    s = sets[("all", k)]
    print(f"A2 top {k}: size {len(s)}, moved vs top 3: out {sorted(base - s)}, in {sorted(s - base)}, share moved {len(base ^ s) / max(1, len(base)):.2f}")
d = used_drive(np.arange(len(ppk))); tot = d.sum()
_, rank = child_set(ALL, d, 3)
non = np.where(rank >= 3)[0]; best_non = non[np.argsort(-d[non])[:len(base)]]
print(f"A3 share of used drive: child set {d[list(base)].sum() / tot:.2f}; best same-size set of non-children {d[best_non].sum() / tot:.2f}",
      "| those non-children:", [f"n{j}(rank {rank[j]})" for j in best_non[:8]])
