"""b14 n33 follow-up: flat warm discs do nothing, the flat-mean patch kills it, blur halves it. Does it need the
warm thing to have internal shading/texture?
PREDICTION (before running): a 40 px warm (red-orange) disc with 3-D shading (radial gradient + highlight) pasted
into 20 low photos gives > 25% of max (median), at least 3x the flat disc; the same shaded disc in gray or blue < 10%.
A warm disc filled with fine noise texture also > 20%."""
from lib4 import *
b, n = 14, 33; MX = 5.64; X, y, ana = data(); A = acts(); rng = np.random.default_rng(0)
v = A[f"max{b}"][:, n].astype(np.float32); pool = np.where(v < np.percentile(v, 40))[0]; BG = np.asarray(X[ana[np.sort(rng.choice(pool, 20, replace=False))]])
yy, xx = np.mgrid[0:224, 0:224]; 
def disc(col, D=40, kind="flat"):
    r = np.sqrt((yy - 112) ** 2 + (xx - 112) ** 2) / (D / 2); m = r <= 1; col = np.array(col, np.float32)
    if kind == "flat": tex = np.ones_like(r)
    elif kind == "shaded": tex = np.clip(np.sqrt(np.clip(1 - r ** 2, 0, 1)) * .9 + .25, 0, 1.4) + 0.8 * np.exp(-(((yy - 112 + D * .2) ** 2 + (xx - 112 + D * .2) ** 2) / (D * .12) ** 2))
    elif kind == "noise": tex = 1 + rng.normal(0, .25, r.shape)
    return m, np.clip(col[None, None] * tex[..., None], 0, 255)
res = {}
for cname, col in (("red-orange", (220, 80, 30)), ("pink", (220, 110, 130)), ("gray", (140, 140, 140)), ("blue", (40, 90, 220))):
    for kind in ("flat", "shaded", "noise"):
        for D in (24, 40, 64):
            m, pix = disc(col, D, kind); I = BG.copy().astype(np.float32); I[:, m] = pix[m]
            z = zmap(b, n, I.astype(np.uint8)); res[(cname, kind, D)] = int(100 * np.median(z[:, 5:9, 5:9].max((1, 2))) / MX)
    print(cname, {f"{k} {D}": res[(cname, k, D)] for k in ("flat", "shaded", "noise") for D in (24, 40, 64)})
