"""Chroma tests on natural images for b3 n623 and b4 n499 (and b1 n365).
1) Mean color (YCbCr) and local contrast in the receptive field at the 100 strongest spots vs 100 most negative spots.
2) Color edits on the top-30 images, response = spatial max after / before (median):
   gray, chroma x0.5, x2, chroma rotated 90/180/270 deg in the (Cb, Cr) plane, keep only Cb (blue-yellow), keep only Cr."""
import sys, os, json, numpy as np
from early_lib import *
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vision2"))
import gallery as G

M = np.array([[.299, .587, .114], [-.168736, -.331264, .5], [.5, -.418688, -.081312]], np.float32)
Mi = np.linalg.inv(M)


def ycc(a): return a.astype(np.float32) @ M.T
def rgb(c): return (c @ Mi.T).clip(0, 255).astype(np.uint8)


def edit(a, kind):
    c = ycc(a); y, cb, cr = c[..., 0], c[..., 1], c[..., 2]
    if kind == "gray": cb, cr = 0 * cb, 0 * cr
    elif kind == "chroma0.5": cb, cr = .5 * cb, .5 * cr
    elif kind == "chroma2": cb, cr = 2 * cb, 2 * cr
    elif kind.startswith("rot"):
        t = np.deg2rad(int(kind[3:])); cb, cr = np.cos(t) * cb - np.sin(t) * cr, np.sin(t) * cb + np.cos(t) * cr
    elif kind == "onlyCb": cr = 0 * cr
    elif kind == "onlyCr": cb = 0 * cb
    elif kind == "lumflat": y = 0 * y + y.mean()      # keep color, remove luminance pattern
    elif kind == "lumblur":
        from PIL import Image, ImageFilter
        y = np.array(Image.fromarray(y.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(4))).astype(np.float32)
    return rgb(np.stack([y, cb, cr], -1))


KINDS = ["gray", "chroma0.5", "chroma2", "rot90", "rot180", "rot270", "onlyCb", "onlyCr", "lumflat", "lumblur"]
if __name__ == "__main__":
    out = {}
    for b, n in [(3, 623), (4, 499), (1, 365)]:
        S = G.scan({b: [n]})[(b, n)]; vm, pm, vn, pn = S; X = G._X; IDX = G._idx
        R = 56 // 2 ** G._m.stage_of[b]; s = 224 // R; cs = G.CROP[G._m.stage_of[b]] // 2

        def spot(i, p):
            r, c = divmod(int(p), R); cy, cx = int((r + .5) * s), int((c + .5) * s)
            a = np.asarray(X[IDX[i]])[max(cy - cs // 2, 0):cy + cs // 2, max(cx - cs // 2, 0):cx + cs // 2]
            c3 = ycc(a); return [float(c3[..., 0].mean()), float(c3[..., 1].mean()), float(c3[..., 2].mean()), float(c3[..., 0].std()),
                                 float(np.hypot(c3[..., 1], c3[..., 2]).mean()), float(np.hypot(c3[..., 1] - c3[..., 1].mean(), c3[..., 2] - c3[..., 2].mean()).mean())]
        top = np.argsort(-vm)[:100]; neg = np.argsort(vn)[:100]; rnd = np.random.default_rng(0).choice(len(vm), 100, replace=False)
        st = {k: np.median([spot(i, (pm if k != "neg" else pn)[i]) for i in ix], 0).round(1).tolist()
              for k, ix in (("top", top), ("neg", neg), ("random", rnd))}
        print(f"\n{b},{n} spot color (median) [Y, Cb(+blue), Cr(+red), Y std, chroma magnitude, chroma spread]")
        for k, v in st.items(): print(f"  {k:6s}", v)
        imgs = np.stack([np.asarray(X[IDX[i]]) for i in top[:30]]); base = zmaps(b, n, imgs).max((1, 2))
        ed = {}
        for k in KINDS:
            ed[k] = float(np.median(zmaps(b, n, np.stack([edit(a, k) for a in imgs])).max((1, 2)) / base))
        print("  edits on top-30:", {k: round(v, 2) for k, v in ed.items()})
        out[f"{b},{n}"] = {"spot": st, "edits": ed}
    json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)

