"""2 n332 (both poles are flat-looking crops). At its top-30 and bottom-30 spots (3000-image set):
position stats, local patch contrast, and edits: flatten window (24 px around spot -> mean), keep only window
(outside -> window mean), blur 1, add faint noise (sd 3), JPEG q30, shift 2 px / 4 px. Value at the same spot."""
import sys, os, io, json, numpy as np
from PIL import Image, ImageFilter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from early_lib import *
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "vision2"))
import gallery as G
T = (2, 332); G._init(); S = G.scan({2: [332]}); vm, pm, vn, pn = S[T]; R = 56
def edits(img, r, c):
    cy, cx = int((r + .5) * 4), int((c + .5) * 4); out = {"orig": img}
    a = img.copy().astype(np.float32); t, b, l, rr = max(cy - 12, 0), cy + 12, max(cx - 12, 0), cx + 12
    a[t:b, l:rr] = a[t:b, l:rr].mean((0, 1)); out["flatten24"] = a
    k = np.tile(img[t:b, l:rr].astype(np.float32).mean((0, 1)), (224, 224, 1)); k[t:b, l:rr] = img[t:b, l:rr]; out["only24"] = k
    out["blur1"] = np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(1)))
    out["noise3"] = img + np.random.default_rng(0).normal(0, 3, img.shape)
    bio = io.BytesIO(); Image.fromarray(img).save(bio, "JPEG", quality=30); out["jpeg30"] = np.array(Image.open(bio))
    out["shift2"] = np.roll(img, 2, 1); out["shift4"] = np.roll(img, 4, 1); out["gray"] = np.repeat(img.mean(2, keepdims=True), 3, 2)
    return {k: np.clip(v, 0, 255).astype(np.uint8) for k, v in out.items()}
res = {}
for name, order, P in (("top", np.argsort(-vm)[:30], pm), ("neg", np.argsort(vn)[:30], pn)):
    names = None; vals = []; info = []
    for i in order:
        r, c = divmod(int(P[i]), R); img = np.asarray(G._X[G._idx[i]]); E = edits(img, r, c); names = list(E)
        Z = zmaps(2, 332, np.stack(list(E.values()))); vals.append(Z[:, r, c])
        cy, cx = int((r + .5) * 4), int((c + .5) * 4); w = img[max(cy - 12, 0):cy + 12, max(cx - 12, 0):cx + 12].astype(np.float32)
        info.append((r, c, w.mean(), w.std()))
    V = np.array(vals); base = V[:, 0:1]
    print(f"\n{name}: orig median {np.median(base):+.2f}; positions r/c median {np.median([x[0] for x in info]):.0f}/{np.median([x[1] for x in info]):.0f}"
          f" border(<3 or >52) frac {np.mean([min(x[0], x[1]) < 3 or max(x[0], x[1]) > 52 for x in info]):.2f}; window mean {np.median([x[2] for x in info]):.0f} sd {np.median([x[3] for x in info]):.1f}")
    print("  edit / orig (median):", ", ".join(f"{n} {np.median(V[:, j] / base[:, 0]):+.2f}" for j, n in enumerate(names)))
    res[name] = {"names": names, "vals": V.tolist(), "info": [list(map(float, x)) for x in info]}
json.dump(res, open(__file__.replace(".py", ".json"), "w"))
