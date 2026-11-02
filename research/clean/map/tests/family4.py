"""Parent / child check for the 4 study neurons. For each relative R of child C:
 rate   = P(R on) over 400 random images x positions (aligned to the 14x14 grid; finer maps max-pooled, 7x7 maps
          upsampled), on = z > 0
 R|C    = P(R on | C on)   (high + R broader = R's region contains C's)     C|R = P(C on | R on)
 cover  = R's z at C's peak cell on C's top-24 images, median, as % of R's own max (from the deep json)
 sig    = R's response to the child's signature manipulation (same as used in C's test), see SIG below."""
from lib4 import *
import json
from PIL import Image, ImageDraw
X, y, ana = data(); A = acts(); rng = np.random.default_rng(0)
R14 = np.asarray(X[np.sort(rng.choice(ana, 400, replace=False))])
DEEP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deep")
def mx_of(b, n):
    f = os.path.join(DEEP, f"b{b}_n{n}.json"); return json.load(open(f))["max"] if os.path.exists(f) else None
def to14(z):
    R = z.shape[1]
    if R == 14: return z
    if R > 14: k = R // 14; return z.reshape(len(z), 14, k, 14, k).max((2, 4))
    return z.repeat(14 // R, 1).repeat(14 // R, 2)
def hr(a, d):
    h = np.array(Image.fromarray(a).convert("HSV")); h[..., 0] = ((h[..., 0].astype(int) + int(d / 360 * 256)) % 256).astype(np.uint8); return np.array(Image.fromarray(h, "HSV").convert("RGB"))
def arcs(N=256, s=0):
    r = np.random.default_rng(s); P = Image.new("RGB", (224, 224), (190, 190, 190)); D = ImageDraw.Draw(P)
    for _ in range(N):
        x, yy = r.uniform(-10, 224, 2); a = r.uniform(0, 360); D.arc([x, yy, x + 30, yy + 30], a, a + 200, fill=(40, 40, 40), width=4)
    return np.array(P)
def half(theta):
    import math
    yy, xx = np.mgrid[0:224, 0:224] - 111.5; m = (xx * math.sin(math.radians(theta)) - yy * math.cos(math.radians(theta))) > 0
    return np.where(m[..., None], 200, 70).astype(np.uint8).repeat(3, 2)
FAM = {
 (12, 136): [(9, 1137), (9, 301), (4, 383), (0, 106), (8, 827), (2, 133), (11, 629), (9, 801), (15, 620), (16, 2246), (17, 713)],
 (14, 1326): [(13, 78), (13, 170), (13, 531), (13, 835), (12, 992), (13, 1300), (12, 342), (13, 73), (16, 81), (16, 2083), (17, 1653)],
 (14, 33): [(12, 548), (13, 156), (0, 346), (0, 279), (0, 286), (12, 1505), (3, 169), (13, 983), (16, 517), (17, 819), (17, 2254)],
 (14, 831): [(0, 346), (13, 1318), (5, 233), (0, 58), (0, 24), (0, 268), (3, 92), (13, 1376), (10, 1365), (15, 2323), (16, 2246), (17, 2426)],
}
def sig(child, b, n, Tc):
    if child == (12, 136):
        I = np.stack([arcs(s=s) for s in range(4)]); z0, z1 = zmap(b, n, I).max((1, 2)).mean(), zmap(b, n, 255 - I).max((1, 2)).mean()
        return f"dark arcs {z0:5.2f} / inverted {z1:5.2f}"
    if child == (14, 1326):
        I = np.stack([half(45), half(135), half(0), half(90)]); z = zmap(b, n, I)[:, 2:-2, 2:-2].max((1, 2)) if zmap(b, n, I).shape[1] >= 7 else None
        z = zmap(b, n, I); c = z.shape[1] // 7; z = z[:, c:-c, c:-c].max((1, 2)); return f"oblique {z[:2].mean():5.2f} / straight {z[2:].mean():5.2f}"
    base = zmap(b, n, Tc).max((1, 2))
    if child == (14, 33):
        g = np.stack([np.array(Image.fromarray(t).convert("L").convert("RGB")) for t in Tc]); h = np.stack([hr(t, 120) for t in Tc])
    else:
        g = np.stack([np.array(Image.fromarray(t).convert("L").convert("RGB")) for t in Tc]); h = np.stack([hr(t, 90) for t in Tc])
    rg = np.median(zmap(b, n, g).max((1, 2)) / np.maximum(base, 1e-3)); rh = np.median(zmap(b, n, h).max((1, 2)) / np.maximum(base, 1e-3))
    return f"on child-top: gray {rg:5.2f} hue-rot {rh:5.2f} (base med {np.median(base):.2f})"
for C, rels in FAM.items():
    bc, nc = C; zc = zmap(bc, nc, R14); onc = zc > 0
    top = np.argsort(-A[f"max{bc}"][:, nc].astype(np.float32))[:24]; Tc = np.asarray(X[ana[top]]); Zc = zmap(bc, nc, Tc); pk = [np.unravel_index(np.argmax(z), z.shape) for z in Zc]
    print(f"\n=== child b{bc} n{nc}: rate {onc.mean():.3f};  child sig: {sig(C, bc, nc, Tc[:20])}")
    for (b, n) in rels:
        zr = to14(zmap(b, n, R14)); on = zr > 0; m = mx_of(b, n)
        Zr = to14(zmap(b, n, Tc)); cov = np.median([Zr[k][p] for k, p in enumerate(pk)])
        print(f" b{b:2d} n{n:4d}: rate {on.mean():.3f}  R|C {on[onc].mean():.2f}  C|R {onc[on].mean():.2f}  cover {cov:5.2f} ({'' if m is None else int(100*cov/m)}%)  sig: {sig(C, b, n, Tc[:20])}", flush=True)
