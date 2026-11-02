"""T19: on each anchor's top-200 images (10k set), each candidate's value AT the anchor's peak cell, relative to the
candidate's own max in that image (median), and P(candidate > 0 at the anchor's peak). Near 1 = the anchor's spot is
the candidate's hottest spot (twin); 0.3-0.7 and on = inside the candidate's region but not its core; < 0 = off side."""
from mid_common import *
G._init(); ana = np.load(os.path.join(OUTD, "t2_ana10.npy"))
FAM = {(11, 1031): [(9, 1226), (9, 80), (10, 1172), (10, 1029), (9, 1244), (16, 630)],
       (10, 1316): [(1, 80), (0, 106), (9, 686), (5, 221), (4, 721), (9, 916), (14, 1165)],
       (7, 604): [(6, 840), (6, 1121), (5, 15), (1, 80), (9, 1080), (16, 579)],
       (8, 4): [(6, 205), (5, 652), (5, 642), (6, 459), (7, 370), (7, 570), (6, 1032), (11, 676), (11, 745)]}


def to14(z):
    s = z.shape[1]
    if s > 14: k = s // 14; return z.reshape(len(z), 14, k, 14, k).max((2, 4))
    if s < 14: return z.repeat(2, 1).repeat(2, 2)
    return z


for a, cands in FAM.items():
    z = np.load(os.path.join(OUTD, f"t2_z_{a[0]}_{a[1]}.npy")).astype(np.float32).reshape(-1, 196); top = np.argsort(-z.max(1))[:200]
    pk = z[top].argmax(1); Z = zmaps(np.asarray(G._X[ana[top]]), cands, bs=25); print(f"\nanchor {a}")
    for c in cands:
        zc = to14(Z[c]).reshape(200, -1); v = zc[np.arange(200), pk]; mx_ = zc.max(1)
        print(f"  {str(c):11s} value at anchor peak / own image max: median {np.median(v / np.maximum(mx_, 1e-3)):5.2f}   P(on at peak) {np.mean(v > 0):.2f}   P(on anywhere) {np.mean(mx_ > 0):.2f}")
