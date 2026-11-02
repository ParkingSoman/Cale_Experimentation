"""T16: family check by position-level containment, 1000 random images of the 10k set. All maps pooled to 14x14
(max-pool for finer maps, nearest upsample for 7x7). For each anchor A and candidate X:
  onX|A+  = P(X > 0 | A in its top-1% cells), base = P(X > 0) overall  -> X on wherever A is strongly on? (parent)
  onA|X+  = P(A > 0 | X in its top-1% cells), baseA                    -> A on wherever X is strongly on? (child/twin)
  negX|A+ = P(X < its 5th percentile | A top-1%)                        -> A sits on X's off side?
  r       = Pearson r of relu maps. Plus X's mean z in row 1 vs row 12 (frame code?)."""
from mid_common import *
G._init(); ana = np.load(os.path.join(OUTD, "t2_ana10.npy")); rng = np.random.default_rng(5)
pick = np.sort(rng.choice(len(ana), 1000, replace=False)); imgs = np.asarray(G._X[ana[pick]])
FAM = {(7, 604): [(6, 840), (6, 1121), (5, 15), (5, 591), (6, 1329), (1, 80), (4, 146), (9, 1080), (9, 1233), (16, 579), (6, 1472)],
       (8, 4): [(5, 652), (5, 642), (6, 205), (6, 459), (7, 370), (7, 570), (6, 1032), (11, 676), (11, 745), (11, 318)],
       (10, 1316): [(1, 80), (0, 106), (5, 221), (9, 686), (9, 1304), (4, 721), (9, 916), (16, 1301), (16, 1034), (14, 1165), (1, 263)],
       (11, 1031): [(9, 1226), (9, 80), (10, 1029), (9, 1244), (9, 762), (10, 1172), (9, 56), (16, 630), (16, 2231), (16, 2441), (9, 1506)]}
alln = sorted(set(list(FAM) + [x for v in FAM.values() for x in v]))
Z = zmaps(imgs, alln, bs=25)


def to14(z):
    s = z.shape[1]
    if s == 14: return z
    if s > 14: k = s // 14; return z.reshape(len(z), 14, k, 14, k).max((2, 4))
    return z.repeat(2, 1).repeat(2, 2)


Z = {k: to14(v).reshape(-1) for k, v in Z.items()}
for a, cands in FAM.items():
    za = Z[a]; Ap = za > np.quantile(za, .99); print(f"\nanchor {a}: base P(A>0)={np.mean(za > 0):.3f}")
    print("  cand        onX|A+   baseX   onA|X+   negX|A+     r     row1   row12")
    for x in cands:
        zx = Z[x]; Xp = zx > np.quantile(zx, .99); zr = zx.reshape(-1, 14, 14)
        r = np.corrcoef(np.maximum(za, 0), np.maximum(zx, 0))[0, 1]
        print(f"  {str(x):11s} {np.mean(zx[Ap] > 0):6.2f}  {np.mean(zx > 0):6.2f}   {np.mean(za[Xp] > 0):6.2f}   {np.mean(zx[Ap] < np.quantile(zx, .05)):6.2f}  {r:6.2f}  {zr[:, 1].mean():6.2f}  {zr[:, 12].mean():6.2f}")
