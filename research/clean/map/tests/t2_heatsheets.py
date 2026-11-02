"""T2: heat maps on top-16 and most-negative-8 full images (analysis set, 20/class = 20k images), plus plain gray."""
from mid_common import *
G._init(); _, y = G._X, G._y; ana, _ = split(y); ana = ana[::2]   # 10/class = 10k images
names = [s.split(",")[0] for s in class_names()]
Z = zmaps(G._X[ana]); np.save(os.path.join(OUTD, "t2_ana10.npy"), ana)
for t, z in Z.items():
    np.save(os.path.join(OUTD, f"t2_z_{t[0]}_{t[1]}.npy"), z.astype(np.float16))
    f = z.reshape(len(z), -1); mxv, mnv = f.max(1), f.min(1); cells = []
    for i in np.argsort(-mxv)[:24]: cells.append((heat(G._X[ana[i]], z[i]), f"+{mxv[i]:.1f} {names[y[ana[i]]]}"))
    for i in np.argsort(mnv)[:8]: cells.append((heat(G._X[ana[i]], z[i]), f"{mnv[i]:.1f} {names[y[ana[i]]]}"))
    g = np.full((1, 224, 224, 3), 124, np.uint8); zg = zmaps(g, [t])[t][0]
    cells.append((heat(g[0], zg), f"gray max {zg.max():.2f} min {zg.min():.2f}"))
    print(t, "gray map row12", zg[12].round(2), "row6", zg[6].round(2))
    print(sheet(cells, f"t2_heat_b{t[0]}_n{t[1]}.png", title=f"block {t[0]} n{t[1]} top-24 (10k imgs) | 8 most negative | plain gray"))
