"""14_1043: does darkening toward the frame edge (lens vignetting) drive it? 200 random images: add vignette
(gain falls from 1 at centre to v at corners). Top-20 images: remove it (divide by the same falloff). Also blur-only
border (Gaussian blur on outer 40 px). Reports median image max / neuron max and median border-ring max."""
import sys, os, json, numpy as np
from PIL import Image, ImageFilter
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "s6")); sys.path.insert(0, os.path.join(H, "..", "..", "..", "vision2"))
from judge_lib import maps
from convnext import load_images
key = "14_1043"; C = json.load(open(os.path.join(H, "..", "..", "s8", "candidates.json")))[key]
cs = ["C8", "C9", "C7", "C5", "C1"]; tg = [(14, 1043)] + [tuple(C[c]) for c in cs]
MX = {t: json.load(open(os.path.join(H, "..", "..", "deep", f"b{t[0]}_n{t[1]}.json")))["max"] for t in tg}
D = np.load(os.path.join(H, "out", "scan.npz")); X, _ = load_images(); idx = D["idx"]
v = D[key].astype(np.float32).reshape(len(idx), -1).max(1); top = np.argsort(-v)[:20]
rnd = np.random.default_rng(1).choice(len(idx), 200, replace=False)
yy, xx = np.mgrid[:224, :224]; r = np.sqrt((yy - 111.5) ** 2 + (xx - 111.5) ** 2) / 158.4
def vig(a, vv): return (1 - (1 - vv) * r ** 2)[None, :, :, None]
I = np.stack([np.asarray(X[idx[i]]) for i in top]).astype(np.float32); R = np.stack([np.asarray(X[idx[i]]) for i in rnd]).astype(np.float32)
def bblur(a):
    out = []
    for im in a.astype(np.uint8):
        b = np.asarray(Image.fromarray(im).filter(ImageFilter.GaussianBlur(4))); m = (np.minimum.reduce([yy, xx, 223 - yy, 223 - xx]) < 40)[..., None]
        out.append(np.where(m, b, im))
    return np.stack(out)
E = {"top20 orig": I, "top20 de-vignette .6": I / vig(I, .6), "top20 brighten-only edges x1.3": I * (1 + .3 * r ** 2)[None, :, :, None],
     "rand200 orig": R, "rand200 vignette .6": R * vig(R, .6), "rand200 vignette .3": R * vig(R, .3), "rand200 border blur": bblur(R)}
E = {k: np.clip(a, 0, 255).astype(np.uint8) for k, a in E.items()}
Z = maps(np.concatenate(list(E.values())), tg); o = 0
ring = (np.minimum.reduce(np.mgrid[:14, :14].tolist() + [13 - np.mgrid[:14, :14][0], 13 - np.mgrid[:14, :14][1]]) == 0)
print("edit".ljust(32) + "".join(f"{'T' if j == 0 else cs[j-1]:>14s}" for j in range(len(tg))) + "   (image max | ring max)")
for name, arr in E.items():
    n = len(arr); s = name.ljust(32)
    for t in tg:
        z = Z[t][o:o + n] / MX[t]; s += f"  {np.median(z.reshape(n, -1).max(1)):+.2f}|{np.median(z[:, ring].max(1)):+.2f}"
    print(s); o += n
