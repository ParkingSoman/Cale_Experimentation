"""(a) b3 n623: isoluminant color noise vs luminance noise, grain 2/4/8/16 px, amplitude 20/40/80, axis by/rg.
(b) b4 n499: what color structure it needs. Top-30 natural images: gray / chroma set to the image mean (no color
pattern) / only coarse chroma kept (fine chroma removed) / only fine chroma kept. Synthetic: luminance noise on
a split ground (blue half | yellow half) vs uniform grounds."""
import sys, os, json, numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from early_lib import *
from t_chroma import ycc, rgb
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vision2"))
import gallery as G

rng = np.random.default_rng(2); out = {}


def noise(g):
    return np.array(Image.fromarray(rng.choice([0, 255], (224 // g, 224 // g)).astype(np.uint8)).resize((224, 224), Image.NEAREST)) / 127.5 - 1


stim, lab = [], []
for g in (2, 4, 8, 16):
    s = noise(g)
    for amp in (20, 40, 80):
        for ax in ("lum", "by", "rg"):
            c = np.zeros((224, 224, 3), np.float32); c[..., 0] = 128
            c[..., {"lum": 0, "by": 1, "rg": 2}[ax]] += amp * s; stim.append(rgb(c)); lab.append((g, amp, ax))
r = stat(multi([(3, 623)], np.stack(stim))[(3, 623)]) / 3.01
print("3,623 noise (fraction of max 3.01): grain amp -> lum by rg")
for i in range(0, len(lab), 3): print("  ", lab[i][:2], " ".join(f"{x:6.2f}" for x in r[i:i + 3]))
out["623_noise"] = {"labels": lab, "resp": r.tolist()}

# (b) n499 natural
vm, pm, vn, pn = G.scan({4: [499]})[(4, 499)]; ix = np.argsort(-vm)[:30]
imgs = np.stack([np.asarray(G._X[G._idx[i]]) for i in ix]); base = zmaps(4, 499, imgs).max((1, 2))


def ed(a, k):
    c = ycc(a)
    for ch in (1, 2):
        lo = gaussian_filter(c[..., ch], 6); mean = c[..., ch].mean()
        if k == "gray": c[..., ch] = 0
        elif k == "meanchroma": c[..., ch] = mean
        elif k == "coarseonly": c[..., ch] = lo
        elif k == "fineonly": c[..., ch] = c[..., ch] - lo + mean
    return rgb(c)


res = {k: float(np.median(zmaps(4, 499, np.stack([ed(a, k) for a in imgs])).max((1, 2)) / base)) for k in ("gray", "meanchroma", "coarseonly", "fineonly")}
print("4,499 natural top-30:", {k: round(v, 2) for k, v in res.items()}); out["499_natural"] = res
# (b) synthetic split grounds with luminance noise
s = noise(2); syn, slab = [], []
for name, left, right in (("gray|gray", 0, 0), ("yel|yel", -30, -30), ("blue|blue", 30, 30), ("blue|yel", 30, -30), ("yel|gray", -30, 0), ("blue|gray", 30, 0)):
    for amp in (25, 50):
        c = np.zeros((224, 224, 3), np.float32); c[..., 0] = 128 + amp * s; c[:, :112, 1] = left; c[:, 112:, 1] = right
        syn.append(rgb(c)); slab.append((name, amp))
Z = multi([(4, 499)], np.stack(syn))[(4, 499)]
print("4,499 synthetic luminance noise on grounds (fraction of max 4.51): central max, max at the boundary column")
for i, l in enumerate(slab):
    print("  ", l, f"{Z[i][5:23, 5:23].max() / 4.51:6.2f} {Z[i][5:23, 13:15].max() / 4.51:6.2f}")
out["499_synth"] = {"labels": slab, "center": (Z[:, 5:23, 5:23].max((1, 2)) / 4.51).tolist(), "boundary": (Z[:, 5:23, 13:15].max((1, 2)) / 4.51).tolist()}
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
