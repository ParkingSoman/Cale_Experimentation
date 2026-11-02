"""Patch transplant (for b14 n33 and b14 n831; flat discs / flat fields failed for both).
Cut the 56x56 patch centred on the peak of each top-20 image, paste it at the centre of 20 low-response photos.
Variants: as-is, gray, flat (patch mean colour), shuffled 4x4 sub-tiles (14 px: texture kept, shape broken),
hue+180, blurred (gauss 3). Value = median over 20 of the centre-4x4 max, % of max.
PREDICTIONS (before running):
 n33: as-is > 40%; gray < 60% of as-is; flat < 10%; shuffled keeps > 60% of as-is (small texture-like).
 n831: as-is > 40%; gray < 10%; hue+180 < 20%; shuffled keeps > 60% of as-is (colour texture, not shape)."""
from lib4 import *
from PIL import Image, ImageFilter
X, y, ana = data(); A = acts(); rng = np.random.default_rng(0); S = 56
def variants(p):
    g = np.array(Image.fromarray(p).convert("L").convert("RGB")); f = np.broadcast_to(p.reshape(-1, 3).mean(0), p.shape).astype(np.uint8)
    t = p.reshape(4, 14, 4, 14, 3).transpose(0, 2, 1, 3, 4).reshape(16, 14, 14, 3)[rng.permutation(16)].reshape(4, 4, 14, 14, 3).transpose(0, 2, 1, 3, 4).reshape(S, S, 3)
    h = np.array(Image.fromarray(p).convert("HSV")); h[..., 0] = (h[..., 0].astype(int) + 128) % 256; h = np.array(Image.fromarray(h.astype(np.uint8), "HSV").convert("RGB"))
    bl = np.array(Image.fromarray(p).filter(ImageFilter.GaussianBlur(3)))
    return {"as-is": p, "gray": g, "flat": f, "shuffled": t, "hue+180": h, "blur3": bl}
for b, n, MX in ((14, 33, 5.64), (14, 831, 5.79)):
    v = A[f"max{b}"][:, n].astype(np.float32); top = np.argsort(-v)[:20]; T = np.asarray(X[ana[top]]); Zt = zmap(b, n, T)
    pool = np.where(v < np.percentile(v, 40))[0]; BG = np.asarray(X[ana[np.sort(rng.choice(pool, 20, replace=False))]])
    print(f"b{b} n{n}: backgrounds alone centre {int(100*np.median(zmap(b, n, BG)[:, 5:9, 5:9].max((1,2)))/MX)}%")
    res = {}
    for k in range(20):
        r, c = np.unravel_index(np.argmax(Zt[k]), Zt[k].shape); cy, cx = int((r + .5) * 16), int((c + .5) * 16)
        t0, l0 = np.clip(cy - S // 2, 0, 224 - S), np.clip(cx - S // 2, 0, 224 - S); p = T[k][t0:t0 + S, l0:l0 + S].copy()
        for name, q in variants(p).items():
            im = BG[k].copy(); im[84:84 + S, 84:84 + S] = q; res.setdefault(name, []).append(im)
    for name, I in res.items():
        print(f"   {name:9s} {int(100*np.median(zmap(b, n, np.stack(I))[:, 5:9, 5:9].max((1,2)))/MX):4d}%")
