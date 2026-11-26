"""Checks on "Children, the essay's way" (advisor review, 2026-11-22). Mac, numpy only.
C1 Is the flow share just s3:69's fixed weight row? Correlate mean share with the positive weight row.
C2 Dependence (essay line 231, the right direction): receivers in their OWN strong half at the child cell
   (A_pre at the cell >= half of the receiver's own max peak), and c69 / pre = the parent's part of its input there.
C3 Coverage picked on even rows, scored on odd rows (and the reverse).
C4 P3 controls: shuffle facet labels within animal / artifact; only the 149 photos strong for both; cell match.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
S1 = np.load(os.path.join(R, "oc_step1.npz"))
W = np.load(os.path.join(R, "oc_weights.npz"))
ppk = S1["ppk"].astype(np.float32)
st = np.where(ppk >= .5 * ppk.max())[0]
c = S1["A_c69"][st].astype(np.float32)
pre = S1["A_pre"][st].astype(np.float32)
apos = S1["A_pos"].astype(np.float32)
own_max = apos.max(0)                                          # each receiver's own max peak over 19,000 photos
rng = np.random.default_rng(0)

# C1
on = (pre > 0)
SH = np.clip(c, 0, None) * on; SH = SH / SH.sum(1, keepdims=True)
w69 = np.clip(W["read_all_A"][:, 69].sum((1, 2)), 0, None)
print("C1 corr(mean flow share, positive weight row of s3:69):", round(float(np.corrcoef(SH.mean(0), w69)[0, 1]), 3))
print("   share of photos where each receiver is on: median", round(float(np.median(on.mean(0))), 2))

# C2 dependence
strongrec = pre >= .5 * own_max[None, :]
dep = np.where(strongrec & (pre > 0), np.clip(c, 0, None) / np.maximum(pre, 1e-6), 0)
n_strong = strongrec.sum(1)
print(f"\nC2 receivers in their own strong half at the cell, per strong parent photo: median {int(np.median(n_strong))}, "
      f"mean {n_strong.mean():.1f}, photos with none {(n_strong == 0).mean():.2f}")
cnt = strongrec.sum(0); dm = np.where(strongrec, dep, np.nan)
mdep = np.nanmean(dm, 0)
order = np.argsort(-cnt)
print("   receivers most often strong on parent-strong photos (count of 393; mean c69/pre when strong):")
print("   " + " ".join(f"n{j}:{cnt[j]}/{mdep[j]:.2f}" for j in order[:20]))
dep_order = [j for j in np.argsort(-np.nan_to_num(mdep)) if cnt[j] >= 10][:15]
print("   highest dependence (at least 10 photos): " + " ".join(f"n{j}:{mdep[j]:.2f}({cnt[j]})" for j in dep_order))
# per photo: does one receiver depend mostly on the parent?
best = dep.max(1)
print(f"   per photo, the receiver that depends most on the parent: median c69/pre {np.median(best):.2f}; "
      f"photos where some strong receiver gets at least half its input from s3:69: {(best >= .5).mean():.2f}")
# group structure in the dependence view
lab = S1["lab"][st] < 398  # labels: record-only (split WITHDRAWN 2026-11-23, LOG "Withdrawn")
for nm, m in [("animal", lab), ("artifact", ~lab)]:  # labels: record-only
    cm = strongrec[m].mean(0); top = np.argsort(-cm)[:8]
    print(f"   {nm} photos ({m.sum()}): receivers strong most often " + " ".join(f"n{j}:{cm[j]:.2f}" for j in top))

# C3 coverage picked on one half, scored on the other
h = [np.arange(0, len(st), 2), np.arange(1, len(st), 2)]
for a, b in ((0, 1), (1, 0)):
    o = np.argsort(-SH[h[a]].mean(0))
    print(f"\nC3 pick on half {a}, score on half {b}: " + " ".join(f"top{n}:{(SH[h[b]][:, o[:n]].sum(1) >= .5).mean():.2f}" for n in (20, 25, 30, 40)))
oa, ob = np.argsort(-SH[h[0]].mean(0))[:30], np.argsort(-SH[h[1]].mean(0))[:30]
print("   top-30 sets from the two halves, Jaccard:", round(len(set(oa) & set(ob)) / len(set(oa) | set(ob)), 2))

# C4 P3 controls
Fc = np.load(os.path.join(R, "oc_facets.npz")); FR = np.load(os.path.join(R, "oc_facets_read.npz")); RT = np.load(os.path.join(R, "oc_route.npz"))
pk = Fc["pk"].astype(np.float32); sn = FR["st"]; fa = FR["a4"]
ap = RT["a4"]; SHp = RT["SH"].astype(np.float32); SHp /= np.linalg.norm(SHp, axis=1, keepdims=True) + 1e-9
cent = np.stack([SHp[ap == g].mean(0) for g in range(4)]); cent /= np.linalg.norm(cent, axis=1, keepdims=True)
C = np.clip(S1["A_c69"][sn].astype(np.float32), 0, None) * (S1["A_pre"][sn].astype(np.float32) > 0)
ok = C.sum(1) > 0
PR = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
pg = (PR @ cent.T).argmax(1)
an = S1["lab"][sn] < 398  # labels: record-only


def purity(f, p, k=4, kp=4):
    t = np.zeros((k, kp), int)
    for x, y in zip(f, p): t[x, y] += 1
    return (t.max(1) / np.maximum(t.sum(1), 1)), t


pur, t = purity(fa[ok], pg[ok])
within = []
for _ in range(500):
    sf = fa[ok].copy()
    for cls in (True, False):
        idx = np.where(an[ok] == cls)[0]; sf[idx] = rng.permutation(sf[idx])
    within.append(purity(sf, pg[ok])[0].mean())
print(f"\nC4 P3 purity mean {pur.mean():.2f}; shuffled WITHIN animal/artifact: 99th pct {np.quantile(within, .99):.2f}, mean {np.mean(within):.2f}")
both = np.isin(sn, st) & ok
pb, tb = purity(fa[both], pg[both])
print(f"   only photos strong for both ({both.sum()}): purity " + " ".join(f"{x:.2f}" for x in pb) + f", table {tb.tolist()}")
# cell match: parent peak (14x14) vs n101 peak (7x7)
pcell = S1["parg"][sn].astype(int); pr_, pc_ = pcell // 14 // 2, pcell % 14 // 2
ncell = Fc["arg"][sn].astype(int); nr, nc = ncell // 7, ncell % 7
same = (np.minimum(pr_, 6) == nr) & (np.minimum(pc_, 6) == nc)
print(f"   parent's child cell = n101's peak cell on {same.mean():.2f} of n101's strong photos")
f78 = Fc["split"][sn][:, 78].astype(np.float32); f69 = Fc["split"][sn][:, 69].astype(np.float32)
for g in range(4):
    m = fa == g
    print(f"   facet f{g}: s3:69 contribution mean {f69[m].mean():.2f}, s3:78 {f78[m].mean():.2f}, n101 peak {pk[sn][m].mean():.2f}")
