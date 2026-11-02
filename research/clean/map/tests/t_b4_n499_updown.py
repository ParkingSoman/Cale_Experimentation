"""b4 n499: flip L-R kept 0.98 but rot180 0.72 -> test up-down. Natural top-30: vflip, vflip of chroma only,
vflip of luminance only. Synthetic: horizontal color boundary (top colour / bottom colour) with luminance noise;
vertical colour gradients. Also mean Cb above vs below the peak spot in top-100 vs random spots."""
import sys, os, json, numpy as np
from PIL import Image
from early_lib import *
from t_chroma import ycc, rgb
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vision2"))
import gallery as G

out = {}
vm, pm, vn, pn = G.scan({4: [499]})[(4, 499)]; ix = np.argsort(-vm)[:30]
imgs = np.stack([np.asarray(G._X[G._idx[i]]) for i in ix]); base = zmaps(4, 499, imgs).max((1, 2))


def ed(a, k):
    c = ycc(a)
    if k == "vflip": c = c[::-1]
    elif k == "vflip_chroma": c[..., 1:] = c[::-1, :, 1:]
    elif k == "vflip_lum": c[..., 0] = c[::-1, :, 0]
    elif k == "hflip": c = c[:, ::-1]
    return rgb(c)


res = {k: float(np.median(zmaps(4, 499, np.stack([ed(a, k) for a in imgs])).max((1, 2)) / base)) for k in ("hflip", "vflip", "vflip_chroma", "vflip_lum")}
print("natural top-30:", {k: round(v, 2) for k, v in res.items()}); out["natural"] = res
# colour above vs below the peak (36 px windows) for top-200 spots vs random spots
def ab(i, p):
    r, c = divmod(int(p), 28); cy, cx = r * 8 + 4, c * 8 + 4; a = ycc(np.asarray(G._X[G._idx[i]]))
    up = a[max(cy - 36, 0):cy, max(cx - 18, 0):cx + 18]; dn = a[cy:cy + 36, max(cx - 18, 0):cx + 18]
    if len(up) < 8 or len(dn) < 8: return None
    return [up[..., 0].mean() - dn[..., 0].mean(), up[..., 1].mean() - dn[..., 1].mean(), up[..., 2].mean() - dn[..., 2].mean()]
top = [ab(i, pm[i]) for i in np.argsort(-vm)[:200]]; top = np.array([t for t in top if t is not None])
rng = np.random.default_rng(0); rnd = [ab(i, rng.integers(784)) for i in rng.choice(len(vm), 300)]; rnd = np.array([t for t in rnd if t is not None])
neg = [ab(i, pn[i]) for i in np.argsort(vn)[:200]]; neg = np.array([t for t in neg if t is not None])
for k, v in (("top", top), ("neg", neg), ("random", rnd)):
    print(f"  {k:6s} above-minus-below: dY {np.median(v[:,0]):6.1f}  dCb {np.median(v[:,1]):6.1f} (frac>0 {(v[:,1]>0).mean():.2f})  dCr {np.median(v[:,2]):6.1f}")
    out[f"ab_{k}"] = np.median(v, 0).tolist() + [float((v[:, 1] > 0).mean())]
# synthetic: top colour / bottom colour with luminance noise
rs = np.random.default_rng(3); s = np.array(Image.fromarray(rs.choice([0, 255], (112, 112)).astype(np.uint8)).resize((224, 224), Image.NEAREST)) / 127.5 - 1
syn, lab = [], []
for name, top_cb, bot_cb in (("blue/yel", 30, -30), ("yel/blue", -30, 30), ("blue/gray", 30, 0), ("gray/yel", 0, -30), ("gray/blue", 0, 30), ("yel/gray", -30, 0), ("gray/gray", 0, 0)):
    for amp in (0, 40):
        c = np.zeros((224, 224, 3), np.float32); c[..., 0] = 128 + amp * s; c[:112, :, 1] = top_cb; c[112:, :, 1] = bot_cb
        syn.append(rgb(c)); lab.append((name, amp))
    # smooth gradient version
    c = np.zeros((224, 224, 3), np.float32); c[..., 0] = 128 + 40 * s; c[..., 1] = np.linspace(top_cb, bot_cb, 224)[:, None]; syn.append(rgb(c)); lab.append((name + " grad", 40))
Z = multi([(4, 499)], np.stack(syn))[(4, 499)]
print("synthetic (top colour / bottom colour), fraction of max 4.51: central max, max on boundary rows")
for i, l in enumerate(lab): print("  ", l, f"{Z[i][5:23, 5:23].max() / 4.51:6.2f} {Z[i][12:16, 5:23].max() / 4.51:6.2f}")
out["synth"] = {"labels": lab, "center": (Z[:, 5:23, 5:23].max((1, 2)) / 4.51).tolist()}
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
