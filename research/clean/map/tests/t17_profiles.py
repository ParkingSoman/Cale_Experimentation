"""T17: frame-position profiles (mean z per row and per column, 1000 random images; 14x14 pooled) for the frame-
code candidates: 7 n604 and 10 n1316 with their inputs/readers."""
from mid_common import *
G._init(); ana = np.load(os.path.join(OUTD, "t2_ana10.npy")); rng = np.random.default_rng(5)
pick = np.sort(rng.choice(len(ana), 1000, replace=False)); imgs = np.asarray(G._X[ana[pick]])
L = [(7, 604), (6, 840), (6, 1121), (9, 1080), (9, 1233), (10, 1316), (9, 686), (14, 1165), (16, 1301), (16, 1034), (4, 721)]
Z = zmaps(imgs, L, bs=25); np.set_printoptions(precision=2, suppress=True, linewidth=220)
for k in L:
    z = Z[k]; s = z.shape[1]
    if s > 14: z = z.reshape(len(z), 14, s // 14, 14, s // 14).mean((2, 4))
    if s < 14: z = z.repeat(2, 1).repeat(2, 2)
    r = z.mean((0, 2)); c = z.mean((0, 1)); print(f"{str(k):11s} rows {r}\n{'':11s} cols {c}")
