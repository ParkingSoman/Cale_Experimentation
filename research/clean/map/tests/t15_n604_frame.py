"""T15 (7 n604): frame code or scene-bottom content? 400 random images of the 10k set, original vs upside-down (vflip).
PREDICTION (before run): it is mostly a frame-position code: after vflip, mean z at row 12 still exceeds row 1 by
>= 70% of the original gap (0.56 in T11). Also local edits on its top-40 (as T14): noise at peak, darken below/above."""
from mid_common import *
G._init(); ana = np.load(os.path.join(OUTD, "t2_ana10.npy")); rng = np.random.default_rng(2); t = (7, 604)
np.set_printoptions(precision=2, suppress=True, linewidth=220)
pick = np.sort(rng.choice(len(ana), 400, replace=False)); imgs = np.asarray(G._X[ana[pick]])
for nm, ims in (("orig", imgs), ("vflip", imgs[:, ::-1].copy())):
    m = zmaps(ims, [t])[t].mean((0, 2)); print(f"{nm:6s} mean z per row {m}  gap row12-row1 {m[12]-m[1]:.2f}")
z = np.load(os.path.join(OUTD, "t2_z_7_604.npy")).astype(np.float32); f = z.reshape(len(z), -1)
top = np.argsort(-f.max(1))[:40]; base = f[top].max(1); I = np.asarray(G._X[ana[top]]).astype(np.float32)
pr, pc = np.divmod(f[top].argmax(1), 14); cy, cx = ((pr + .5) * 16).astype(int), ((pc + .5) * 16).astype(int)
noise = np.kron(rng.choice([-40., 40.], (24, 24)), np.ones((2, 2)))[..., None]


def ed(kind):
    out = I.copy()
    for j in range(40):
        x0, x1 = max(cx[j] - 56, 0), min(cx[j] + 56, 224)
        if kind == "dark below": y0 = min(cy[j] + 8, 224); out[j, y0:y0 + 48, x0:x1] *= .5
        if kind == "dark above": y1 = max(cy[j] - 8, 0); out[j, max(y1 - 48, 0):y1, x0:x1] *= .5
        if kind == "dark at peak": out[j, max(cy[j] - 24, 0):cy[j] + 24, max(cx[j] - 24, 0):cx[j] + 24] *= .5
        if kind in ("noise at peak", "noise far"):
            yy, xx = (cy[j], cx[j]) if kind == "noise at peak" else (224 - cy[j], 224 - cx[j]); yy = int(np.clip(yy - 24, 0, 176)); xx = int(np.clip(xx - 24, 0, 176))
            out[j, yy:yy + 48, xx:xx + 48] += noise
        if kind == "hline below": y0 = min(cy[j] + 40, 222); out[j, y0:y0 + 2, :] = 0
    return np.clip(out, 0, 255).astype(np.uint8)


for k in ("dark below", "dark above", "dark at peak", "noise at peak", "noise far"):
    zz = zmaps(ed(k), [t])[t].reshape(40, -1); at = zz[np.arange(40), pr * 14 + pc]
    print(f"{k:14s} max ratio {np.median(zz.max(1) / base):.2f}   at original peak cell {np.median(at / base):.2f}")
