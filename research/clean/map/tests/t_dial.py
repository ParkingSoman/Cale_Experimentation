"""Natural-image dials. For each target: top-30 images and most-negative-30 images.
Cb shift (add a uniform blue (+) or yellow (-) tint), luminance-texture strength (scale Y around its local mean),
chroma-texture strength. Response = spatial max (top set) or spatial min (negative set), median ratio to unedited.
Also: 1 n365's z and GELU at b4 n499's peak spots (on-side or off-side?)."""
import sys, os, json, numpy as np, mlx.core as mx, mlx.nn as nn
from PIL import Image, ImageFilter
from early_lib import *
from t_chroma import ycc, rgb
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vision2"))
import gallery as G


def blur(ch, r=6):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(ch.astype(np.float32), r)


def ed(a, kind, v):
    c = ycc(a)
    if kind == "cb": c[..., 1] += v
    elif kind == "lumtex": m = blur(c[..., 0]); c[..., 0] = m + v * (c[..., 0] - m)
    elif kind == "chrtex":
        for k in (1, 2): m = blur(c[..., k]); c[..., k] = m + v * (c[..., k] - m)
    return rgb(c)


DIALS = {"cb": (-40, -20, 0, 20, 40), "lumtex": (0, .5, 1, 2), "chrtex": (0, .5, 1, 2)}
out = {}
for b, n in [(3, 623), (4, 499)]:
    vm, pm, vn, pn = G.scan({b: [n]})[(b, n)]; X, IDX = G._X, G._idx
    res = {}
    for side, ix in (("top", np.argsort(-vm)[:30]), ("neg", np.argsort(vn)[:30])):
        imgs = np.stack([np.asarray(X[IDX[i]]) for i in ix])
        f = (lambda Z: Z.max((1, 2))) if side == "top" else (lambda Z: Z.min((1, 2)))
        base = f(zmaps(b, n, imgs))
        for k, vals in DIALS.items():
            res[f"{side}:{k}"] = [float(np.median(f(zmaps(b, n, np.stack([ed(a, k, v) for a in imgs]))) / base)) for v in vals]
            print(f"{b},{n} {side:3s} {k:6s}", dict(zip(vals, np.round(res[f'{side}:{k}'], 2))), flush=True)
    out[f"{b},{n}"] = res
# n365 at n499's peaks
m = G._m; vm, pm, vn, pn = G.scan({4: [499]})[(4, 499)]; ix = np.argsort(-vm)[:20]
imgs = np.stack([np.asarray(G._X[G._idx[i]]) for i in ix]); z = zmaps(1, 365, imgs)
zs = []
for j, i in enumerate(ix):
    r, c = divmod(int(pm[i]), 28); zs.append(z[j, 2 * r:2 * r + 2, 2 * c:2 * c + 2].mean())
zs = np.array(zs); g = np.array(nn.gelu(mx.array(zs)))
print("1 n365 z at 4 n499 top-20 peaks: median %.2f, frac<0 %.2f, GELU median %.3f" % (np.median(zs), (zs < 0).mean(), np.median(g)))
out["n365_at_n499_peaks"] = {"z": zs.tolist(), "gelu": g.tolist()}
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
