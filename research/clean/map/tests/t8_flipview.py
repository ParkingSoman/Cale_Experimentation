"""T8: view top images and their mirror images with heat, for 11 n1031 (and 7 n604, 10 n1316 for the record)."""
from mid_common import *
G._init(); ana = np.load(os.path.join(OUTD, "t2_ana10.npy")); names = [s.split(",")[0] for s in class_names()]; y = G._y
for t in [(11, 1031)]:
    z = np.load(os.path.join(OUTD, f"t2_z_{t[0]}_{t[1]}.npy")).astype(np.float32); f = z.reshape(len(z), -1)
    top = np.argsort(-f.max(1))[24:40]; imgs = np.asarray(G._X[ana[top]]); fl = imgs[:, :, ::-1].copy()
    Z0 = zmaps(imgs, [t])[t]; Z1 = zmaps(fl, [t])[t]; cells = []
    for j in range(16):
        s = max(abs(Z0[j]).max(), abs(Z1[j]).max())
        cells.append((heat(imgs[j], Z0[j]), f"{Z0[j].max():.1f} {names[y[ana[top[j]]]]}")); cells.append((heat(fl[j], Z1[j]), f"flipped {Z1[j].max():.1f}"))
    print(sheet(cells, f"t8_flip_b{t[0]}_n{t[1]}.png", ncol=8, size=150, title=f"{t}: images ranked 25-40, original | mirrored"))
