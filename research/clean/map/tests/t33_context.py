"""b14 n33: how much context does it read? Transplant the peak-centred patch of each top-20 image at sizes 32/56/96/
136 px into 20 low photos (same position as in the original). Also on the ORIGINAL image: gray only the central
32 px of the peak vs gray everything except it.
PREDICTION (before running): response grows with patch size, 136 px recovers > 70% of the original peak (median);
graying only the central 32 px costs more than graying everything else (colour sits on the small warm part)."""
from lib4 import *
from PIL import Image
b, n = 14, 33; X, y, ana = data(); A = acts(); rng = np.random.default_rng(0)
v = A[f"max{b}"][:, n].astype(np.float32); top = np.argsort(-v)[:20]; T = np.asarray(X[ana[top]]); Zt = zmap(b, n, T)
pool = np.where(v < np.percentile(v, 40))[0]; BG = np.asarray(X[ana[np.sort(rng.choice(pool, 20, replace=False))]])
pk = [np.unravel_index(np.argmax(z), z.shape) for z in Zt]; base = np.array([Zt[k][p] for k, p in enumerate(pk)])
def box(p, S):
    cy, cx = int((p[0] + .5) * 16), int((p[1] + .5) * 16); t0, l0 = np.clip(cy - S // 2, 0, 224 - S), np.clip(cx - S // 2, 0, 224 - S); return t0, l0
for S in (32, 56, 96, 136):
    I = []
    for k, p in enumerate(pk):
        t0, l0 = box(p, S); im = BG[k].copy(); im[t0:t0 + S, l0:l0 + S] = T[k][t0:t0 + S, l0:l0 + S]; I.append(im)
    Z = zmap(b, n, np.stack(I)); print(f"transplant {S:3d}px: z at peak / original median {np.median([Z[k][p] / base[k] for k, p in enumerate(pk)]):.2f}")
G = np.stack([np.array(Image.fromarray(t).convert("L").convert("RGB")) for t in T]); I1, I2 = [], []
for k, p in enumerate(pk):
    t0, l0 = box(p, 32); a = T[k].copy(); a[t0:t0 + 32, l0:l0 + 32] = G[k][t0:t0 + 32, l0:l0 + 32]; I1.append(a)
    c = G[k].copy(); c[t0:t0 + 32, l0:l0 + 32] = T[k][t0:t0 + 32, l0:l0 + 32]; I2.append(c)
for lab, I in (("gray centre 32px only", I1), ("gray all but centre 32px", I2), ("gray all", G)):
    Z = zmap(b, n, np.stack(I)); print(f"{lab:26s}: ratio {np.median([Z[k][p] / base[k] for k, p in enumerate(pk)]):.2f}")
