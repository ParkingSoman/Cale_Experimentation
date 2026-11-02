"""b14 n1326 follow-up. Roll test moved the peak inward, but a wrap roll also makes a seam (a cut) in the middle.
Separate 'content near the old border' from 'a cut edge':
 S1 reflect-shift: shift the top-24 images 56 px down-right and fill the gap by mirror reflection (no seam). If the
    unit tracks content, the peak moves inward with the old border content; if it tracks the frame, it stays on ring 0-1.
 S2 seam only: put two unrelated random images side by side (left half of one, right half of another): does a seam
    in the centre drive the unit above its level on the same halves without a seam?
PREDICTION: S1 peak stays near the new frame (ring 0-1) in >=60% of images (frame); S2 seam adds < 0.5.
Saves heat grids for S1 and the wrap roll."""
from lib4 import *
X, y, ana = data(); A = acts(); rng = np.random.default_rng(1); b, n = 14, 1326; nm = names()
R = 14; ii = np.arange(R); d = np.minimum(np.minimum(ii[:, None], ii[None, :]), np.minimum(R - 1 - ii[:, None], R - 1 - ii[None, :]))
v = A[f"max{b}"][:, n].astype(np.float32); top = np.argsort(-v)[:24]; T = np.asarray(X[ana[top]]); Zt = zmap(b, n, T)
def rshift(im, s):
    p = np.pad(im, ((s, 0), (s, 0), (0, 0)), mode="reflect"); return p[:224, :224]
S = np.stack([rshift(t, 56) for t in T]); Zs = zmap(b, n, S)
pk0 = [np.unravel_index(np.argmax(z), z.shape) for z in Zt]; pk1 = [np.unravel_index(np.argmax(z), z.shape) for z in Zs]
print("S1 reflect-shift: peak ring counts", np.bincount([d[p] for p in pk1], minlength=7), "orig", np.bincount([d[p] for p in pk0], minlength=7))
moved = [ (p1[0]-p0[0], p1[1]-p0[1]) for p0, p1 in zip(pk0, pk1)]
print("   peak displacement (rows, cols) [content shift = +3.5,+3.5 cells]:", moved)
print("   median max ratio", round(float(np.median(Zs.max((1, 2)) / Zt.max((1, 2)))), 2))
# does the response at the shifted location of the original peak hold?
held = [Zs[k][min(p[0] + 4, 13), min(p[1] + 4, 13)] / Zt[k][p] for k, p in enumerate(pk0)]
print("   z at (orig peak + 4 cells) / orig peak: median", round(float(np.median(held)), 2))
grid([overlay(S[k], Zs[k]) for k in range(24)], [nm[y[ana[t]]] for t in top], "b14_n1326_reflectshift_heat.png", title="1326 reflect-shift 56px")
Tr = np.roll(T, (112, 112), axis=(1, 2)); grid([overlay(Tr[k], zmap(b, n, Tr[k:k+1])[0]) for k in range(24)], [nm[y[ana[t]]] for t in top], "b14_n1326_roll_heat.png", title="1326 wrap roll 112")
# S2 seams
r = np.asarray(X[np.sort(rng.choice(ana, 200, replace=False))]); a, c = r[:100], r[100:]
J = a.copy(); J[:, :, 112:] = c[:, :, 112:]
za, zc, zj = zmap(b, n, a), zmap(b, n, c), zmap(b, n, J)
base = np.concatenate([za[:, :, 5:7], zc[:, :, 7:9]], 2).mean(); seam = zj[:, :, 5:9].mean()
print(f"S2 seam: mean z in centre columns without seam {base:.2f}, with seam {seam:.2f}; max over seam cols {zj[:, :, 5:9].max((1,2)).mean():.2f} vs {np.maximum(za[:, :, 5:7].max((1,2)), zc[:, :, 7:9].max((1,2))).mean():.2f}")
