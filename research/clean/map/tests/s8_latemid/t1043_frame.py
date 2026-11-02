"""14_1043 test: is the response tied to the frame edge or to the content there? Top-20 images (3000 set):
orig / shift content 48 px right (gray fill on left) / shrink to 128 px centred on gray / mirror-pad (reflect) shift.
Reports per neuron: median of image max / neuron max; border-ring share of the peak; and for the shift edit the value
at the NEW frame left edge (col 0) vs where the OLD left-edge content now sits (col 3). Synthetic: flat gray, gray +
centred big noise disc, full noise. Candidates scored on the same images."""
import sys, os, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "s6")); sys.path.insert(0, os.path.join(H, "..", "..", "..", "vision2"))
from judge_lib import maps
from convnext import load_images
from PIL import Image
key = "14_1043"; C = json.load(open(os.path.join(H, "..", "..", "s8", "candidates.json")))[key]
cs = ["C8", "C9", "C7", "C5", "C1"]; tg = [(14, 1043)] + [tuple(C[c]) for c in cs]
MX = {t: json.load(open(os.path.join(H, "..", "..", "deep", f"b{t[0]}_n{t[1]}.json")))["max"] for t in tg}
D = np.load(os.path.join(H, "out", "scan.npz")); X, _ = load_images(); idx = D["idx"]
top = np.argsort(-D[key].astype(np.float32).reshape(len(idx), -1).max(1))[:20]; I = np.stack([np.asarray(X[idx[i]]) for i in top])
def shift(im, dx, fill="gray"):
    o = np.full_like(im, 128) if fill == "gray" else np.pad(im, ((0, 0), (dx, 0), (0, 0)), mode="reflect")[:, :224]
    if fill == "gray": o[:, dx:] = im[:, :224 - dx]
    return o
def shrink(im, s=128):
    o = np.full_like(im, 128); p = (224 - s) // 2; o[p:p + s, p:p + s] = np.asarray(Image.fromarray(im).resize((s, s))); return o
rng = np.random.default_rng(0); yy, xx = np.mgrid[:224, :224]
gray = np.full((224, 224, 3), 128, np.uint8); noise = np.clip(rng.normal(128, 50, (224, 224, 3)), 0, 255).astype(np.uint8)
disc = gray.copy(); m = (yy - 112) ** 2 + (xx - 112) ** 2 < 90 ** 2; disc[m] = noise[m]
ring = noise.copy(); ring[m] = 128
E = {"orig": I, "shift48_gray": np.stack([shift(i, 48) for i in I]), "shift48_reflect": np.stack([shift(i, 48, "refl") for i in I]),
     "shrink128": np.stack([shrink(i) for i in I]), "flip": I[:, :, ::-1]}
S = {"flat_gray": gray[None], "gray+noise_disc": disc[None], "full_noise": noise[None], "noise_ring_gray_centre": ring[None]}
allim = np.concatenate(list(E.values()) + list(S.values())); Z = maps(allim, tg); res = {}
def bshare(z):
    a = z.reshape(len(z), -1).argmax(1); r, c = a // 14, a % 14; return float(((r == 0) | (r == 13) | (c == 0) | (c == 13)).mean())
print("edit".ljust(24) + "".join(f"{'T' if j == 0 else cs[j-1]:>16s}" for j in range(len(tg))))
o = 0
for name, arr in list(E.items()) + list(S.items()):
    n = len(arr); row = []
    for t in tg:
        z = Z[t][o:o + n]; row.append((float(np.median(z.reshape(n, -1).max(1)) / MX[t]), bshare(z)))
    res[name] = row; print(name.ljust(24) + "".join(f"  {a:+5.2f} (b{b:.2f})" for a, b in row))
    if name == "shift48_gray":
        for j, t in enumerate(tg):
            z = Z[t][o:o + n] / MX[t]; print(f"     {'T' if j == 0 else cs[j-1]}: newedge col0 {np.median(z[:, :, 0].max(1)):+.2f}  oldedge-content col3 {np.median(z[:, :, 3].max(1)):+.2f}  right col13 {np.median(z[:, :, 13].max(1)):+.2f}")
    if name == "orig":
        for j, t in enumerate(tg):
            z = Z[t][o:o + n] / MX[t]; print(f"     {'T' if j == 0 else cs[j-1]}: col0 {np.median(z[:, :, 0].max(1)):+.2f}  col3 {np.median(z[:, :, 3].max(1)):+.2f}  centre {np.median(z[:, 5:9, 5:9].max((1,2))):+.2f} col13 {np.median(z[:, :, 13].max(1)):+.2f}")
    o += n
json.dump({"cands": cs, "res": res}, open(os.path.join(H, "out", "t1043_frame.json"), "w"), indent=1)
