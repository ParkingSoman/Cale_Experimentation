"""T6: frame-anchored or content-anchored? Top-50 images (10k analysis maps from T2) of 7 n604, 10 n1316, 11 n1031.
Shift content by +-32/64/96 px vertically (mean-color fill), and horizontally for 11 n1031; flip; and a 288-px
canvas (image centred, 32 px mean-color border all round). Report median peak ratio and median peak row/col.
PREDICTIONS (before run):
 7 n604: frame-anchored. Shift content DOWN 64 px -> peak stays in rows 11-12 and keeps >= 70% of its value.
 10 n1316: content-anchored (flanks of a centred subject). Shift down 64 px -> peak row moves down by >= 2 cells.
 11 n1031: object-anchored with a side: after a left-right flip the peak moves to the other half of the frame
           (median peak col goes from <7 to >=7) rather than staying left."""
from mid_common import *
G._init(); ana = np.load(os.path.join(OUTD, "t2_ana10.npy"))
T = [(7, 604), (10, 1316), (11, 1031)]


def shift(im, dy=0, dx=0):
    f = im.reshape(-1, 3).mean(0).astype(np.uint8); out = np.empty_like(im); out[:] = f
    H = 224; ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, H - dx), slice(dx, H)) if dx >= 0 else (slice(-dx, H), slice(0, H + dx))
    out[yd, xd] = im[ys, xs]; return out


def big(im):
    f = im.reshape(-1, 3).mean(0).astype(np.uint8); out = np.empty((288, 288, 3), np.uint8); out[:] = f; out[32:256, 32:256] = im; return out


for t in T:
    z = np.load(os.path.join(OUTD, f"t2_z_{t[0]}_{t[1]}.npy")).astype(np.float32); f = z.reshape(len(z), -1)
    top = np.argsort(-f.max(1))[:50]; imgs = np.asarray(G._X[ana[top]]); base = f[top].max(1); r0, c0 = np.divmod(f[top].argmax(1), 14)
    print(f"\n{t}: base peak median {np.median(base):.2f}, peak row median {np.median(r0):.0f}, col median {np.median(c0):.0f}")
    print("  row hist", np.bincount(r0, minlength=14), " col hist", np.bincount(c0, minlength=14))
    eds = {f"dy{d:+d}": [shift(im, dy=d) for im in imgs] for d in (-96, -64, -32, 32, 64, 96)}
    if t == (11, 1031): eds.update({f"dx{d:+d}": [shift(im, dx=d) for im in imgs] for d in (-64, -32, 32, 64)})
    eds["flip"] = [im[:, ::-1] for im in imgs]
    for k, L in eds.items():
        zz = zmaps(np.stack(L), [t])[t].reshape(50, -1); r, c = np.divmod(zz.argmax(1), 14)
        print(f"  {k:6s} ratio {np.median(zz.max(1) / base):.2f}  peak row {np.median(r):4.1f} col {np.median(c):4.1f}  | row hist {np.bincount(r, minlength=14)}")
    zz = zmaps(np.stack([big(im) for im in imgs]), [t])[t]; ff = zz.reshape(50, -1); r, c = np.divmod(ff.argmax(1), 18)
    print(f"  288-canvas ratio {np.median(ff.max(1) / base):.2f}  peak row {np.median(r):.1f}/18 (content rows 2-15) col {np.median(c):.1f}  row hist {np.bincount(r, minlength=18)}")
