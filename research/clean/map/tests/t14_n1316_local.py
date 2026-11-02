"""T14 (10 n1316), natural-image local edits on its top-40 images (10k set). Bands are 48 px tall, 112 px wide,
centred on the peak column; 'below' starts 8 px under the peak cell centre, 'above' ends 8 px over it.
PREDICTION (before run): darkening the band BELOW the peak (x0.5) drops the response to <= 60%; darkening the band
ABOVE keeps >= 90%; sharp fine noise (+-40, 2 px grain) on a 48 px square at the peak drops it to <= 60%; the same
noise in the opposite corner of the frame keeps >= 90%."""
from mid_common import *
G._init(); ana = np.load(os.path.join(OUTD, "t2_ana10.npy")); rng = np.random.default_rng(4); t = (10, 1316)
z = np.load(os.path.join(OUTD, "t2_z_10_1316.npy")).astype(np.float32); f = z.reshape(len(z), -1)
top = np.argsort(-f.max(1))[:40]; base = f[top].max(1); imgs = np.asarray(G._X[ana[top]]).astype(np.float32)
pr, pc = np.divmod(f[top].argmax(1), 14); cy, cx = ((pr + .5) * 16).astype(int), ((pc + .5) * 16).astype(int)
noise = np.kron(rng.choice([-40., 40.], (24, 24)), np.ones((2, 2)))[..., None]


def ed(kind):
    out = imgs.copy()
    for j in range(40):
        x0, x1 = max(cx[j] - 56, 0), min(cx[j] + 56, 224)
        if kind == "dark below": y0 = min(cy[j] + 8, 224); out[j, y0:y0 + 48, x0:x1] *= .5
        if kind == "dark above": y1 = max(cy[j] - 8, 0); out[j, max(y1 - 48, 0):y1, x0:x1] *= .5
        if kind == "bright above": y1 = max(cy[j] - 8, 0); out[j, max(y1 - 48, 0):y1, x0:x1] = 255 - .5 * (255 - out[j, max(y1 - 48, 0):y1, x0:x1])
        if kind in ("noise at peak", "noise far"):
            yy, xx = (cy[j], cx[j]) if kind == "noise at peak" else (224 - cy[j], 224 - cx[j]); yy = int(np.clip(yy - 24, 0, 176)); xx = int(np.clip(xx - 24, 0, 176))
            out[j, yy:yy + 48, xx:xx + 48] += noise
    return np.clip(out, 0, 255).astype(np.uint8)


for k in ("dark below", "dark above", "bright above", "noise at peak", "noise far"):
    zz = zmaps(ed(k), [t])[t].reshape(40, -1); at = zz[np.arange(40), pr * 14 + pc]
    print(f"{k:14s} max ratio {np.median(zz.max(1) / base):.2f}   value at original peak cell ratio {np.median(at / base):.2f}")
