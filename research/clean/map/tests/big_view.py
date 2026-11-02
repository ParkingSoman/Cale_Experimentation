"""Big views: top-K images over the 20k analysis set, full image with heat + 136px crop. Also most negative.
Usage: python big_view.py B,N [B,N ...]"""
import sys; from lib4 import *
for a in sys.argv[1:]:
    b, n = map(int, a.split(",")); A = acts(); X, y, ana = data(); nm = names()
    for pole, key, sgn in (("pos", f"max{b}", -1), ("neg", f"min{b}", 1)):
        v = A[key][:, n].astype(np.float32); order = np.argsort(sgn * v)[:24]; imgs = np.asarray(X[ana[order]])
        Z = zmap(b, n, imgs); R = Z.shape[1]; tiles = []; labs = []
        for k in range(len(order)):
            z = Z[k]; p = np.unravel_index(np.argmax(z) if pole == "pos" else np.argmin(z), z.shape)
            tiles.append(overlay(imgs[k], z)); labs.append(f"{nm[y[ana[order[k]]]]} {v[order[k]]:.1f}")
        grid(tiles, labs, f"b{b}_n{n}_{pole}_heat.png", title=f"b{b} n{n} {pole}: full image + heat (red=+ blue=-)")
        tiles = []
        for k in range(len(order)):
            z = Z[k]; p = np.unravel_index(np.argmax(z) if pole == "pos" else np.argmin(z), z.shape); tiles.append(crop_at(imgs[k], p, R))
        print(grid(tiles, labs, f"b{b}_n{n}_{pole}_crop.png", title=f"b{b} n{n} {pole}: 136px crop at peak"))
