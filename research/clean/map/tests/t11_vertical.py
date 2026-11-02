"""T11 (7 n604, also 10 n1316): a vertical-position code tied to where the picture CONTENT ends?
PREDICTION (7 n604, before run): over random images, mean z rises with row: row 12 exceeds row 1 by > 1.0.
With the lower half of each image replaced by its mean colour, the row of highest mean z moves up to rows 5-6
(2 cells above the fill). With the upper half replaced, the row of LOWEST mean z moves down to rows 8-9."""
from mid_common import *
G._init(); ana = np.load(os.path.join(OUTD, "t2_ana10.npy")); rng = np.random.default_rng(2)
np.set_printoptions(precision=2, suppress=True, linewidth=220)
for t in [(7, 604), (10, 1316), (8, 4), (11, 1031)]:
    z = np.load(os.path.join(OUTD, f"t2_z_{t[0]}_{t[1]}.npy")).astype(np.float32)
    print(t, "10k mean z per row :", z.mean((0, 2))); print(t, "10k mean z per col :", z.mean((0, 1)))
pick = np.sort(rng.choice(len(ana), 400, replace=False)); imgs = np.asarray(G._X[ana[pick]])


def fill(ims, part):
    out = ims.copy()
    for k, im in enumerate(ims):
        f = im.reshape(-1, 3).mean(0).astype(np.uint8)
        if part == "low": out[k, 112:] = f
        if part == "up": out[k, :112] = f
        if part == "lowq": out[k, 168:] = f
    return out


for t in [(7, 604), (10, 1316)]:
    for part in ("none", "low", "lowq", "up"):
        zz = zmaps(fill(imgs, part) if part != "none" else imgs, [t])[t]; m = zz.mean((0, 2))
        print(f"{t} fill={part:5s} mean z per row: {m}  argmax row {m.argmax()} argmin row {m.argmin()}")
