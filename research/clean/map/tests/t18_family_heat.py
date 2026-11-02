"""T18: heat maps of the anchor and its family candidates on the SAME 6 top images of the anchor (10k set).
Each row = one neuron; red = positive, blue = negative, each map scaled to its own max |z|."""
from mid_common import *
G._init(); ana = np.load(os.path.join(OUTD, "t2_ana10.npy"))
FAM = {(11, 1031): [(9, 1226), (9, 80), (10, 1172), (10, 1029), (16, 630), (16, 2231)],
       (10, 1316): [(1, 80), (9, 686), (5, 221), (4, 721), (14, 1165), (16, 1301)],
       (7, 604): [(6, 840), (6, 1121), (1, 80), (9, 1080), (16, 579)],
       (8, 4): [(6, 205), (5, 652), (6, 459), (7, 570), (11, 676), (11, 745)]}
for a, cands in FAM.items():
    z = np.load(os.path.join(OUTD, f"t2_z_{a[0]}_{a[1]}.npy")).astype(np.float32); top = np.argsort(-z.reshape(len(z), -1).max(1))[[0, 2, 4, 6, 8, 10]]
    imgs = np.asarray(G._X[ana[top]]); L = [a] + cands; Z = zmaps(imgs, L, bs=6); cells = []
    for k in L:
        for j in range(6): cells.append((heat(imgs[j], Z[k][j]), f"{k[0]}n{k[1]} {Z[k][j].max():.1f}"))
    print(sheet(cells, f"t18_fam_b{a[0]}_n{a[1]}.png", ncol=6, size=130, title=f"anchor {a} (row 1) vs family rows"))
