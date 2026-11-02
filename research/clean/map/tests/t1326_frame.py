"""b14 n1326: frame-locked or content-locked?
PREDICTION (written before running): the unit is tied to the image frame. (a) Mean z in the outer ring of the 14x14
map is >2 units above the centre over 500 random images. (b) Rolling the top-24 images by 112 px (wrap) keeps the
peak on the outer ring in >=75% of images, and peak value stays >=60% of the unrolled value. (c) A flat gray image is
higher at the border than the centre. (d) Pasting the top images at half size in a mean-colour frame (zoomout)
moves the peak to the inner object's edge? -- open question, reported only."""
from lib4 import *
X, y, ana = data(); A = acts(); rng = np.random.default_rng(0)
b, n = 14, 1326
def rings(Z):
    R = Z.shape[-1]; ii = np.arange(R); d = np.minimum.reduce([ii[:, None] + 0 * ii, ii[None, :] + 0 * ii[:, None], R - 1 - ii[:, None] + 0 * ii, R - 1 - ii[None, :] + 0 * ii[:, None]])
    return [float(Z[..., d == k].mean()) for k in range(R // 2)], d
rand = np.asarray(X[np.sort(rng.choice(ana, 500, replace=False))]); Zr = zmap(b, n, rand); rr, d = rings(Zr)
print("(a) random 500, mean z by ring (0=outer):", [round(v, 2) for v in rr])
v = A[f"max{b}"][:, n].astype(np.float32); top = np.argsort(-v)[:24]; T = np.asarray(X[ana[top]]); Zt = zmap(b, n, T)
print("    top24, mean z by ring:", [round(v, 2) for v in rings(Zt)[0]])
pk = np.array([d.flat[np.argmax(z)] for z in Zt]); print("    top24 peak ring counts:", np.bincount(pk, minlength=7))
Tr = np.roll(T, (112, 112), axis=(1, 2)); Zrr = zmap(b, n, Tr); pk2 = np.array([d.flat[np.argmax(z)] for z in Zrr])
print("(b) rolled 112: peak ring counts:", np.bincount(pk2, minlength=7), " median max ratio:", round(float(np.median(Zrr.max((1, 2)) / Zt.max((1, 2)))), 2))
print("    rolled: mean z by ring:", [round(v, 2) for v in rings(Zrr)[0]])
for g in (0, 128, 255):
    Zg = zmap(b, n, np.full((1, 224, 224, 3), g, np.uint8))[0]; print(f"(c) flat {g}: ring means", [round(v, 2) for v in rings(Zg)[0]])
# (d) noise image
Zn = zmap(b, n, rng.integers(0, 256, (4, 224, 224, 3)).astype(np.uint8)); print("(c2) uniform noise: ring means", [round(v, 2) for v in rings(Zn)[0]])
