"""Hue-rotation curve on each target's own top-20 images (12_245, 13_404): rotate hue by 0..330 deg (12 steps) and
desaturate (sat x .5, x0). Score = median over images of the response at the target's own peak cell / at orig.
Candidates read at the same cell (early-block maps max-pooled to 14x14). r = correlation of the 14-point curve with
the target's. Usage: python t_huecurve.py"""
import sys, os, json, numpy as np
from PIL import Image
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "s6")); sys.path.insert(0, os.path.join(H, "..", "..", "..", "vision2"))
from judge_lib import maps
from convnext import load_images
C = json.load(open(os.path.join(H, "..", "..", "s8", "candidates.json"))); D = np.load(os.path.join(H, "out", "scan.npz")); X, _ = load_images(); idx = D["idx"]
def hsv_edit(im, dh, sm):
    h = np.asarray(Image.fromarray(im).convert("HSV")).astype(np.int32); h[..., 0] = (h[..., 0] + int(dh / 360 * 256)) % 256
    h[..., 1] = np.clip(h[..., 1] * sm, 0, 255); return np.asarray(Image.fromarray(h.astype(np.uint8), "HSV").convert("RGB"))
steps = [(d, 1.0) for d in range(0, 360, 30)] + [(0, .5), (0, 0.)]
for key in ["12_245", "13_404"]:
    tb, tn = map(int, key.split("_")); tg = [(tb, tn)] + [tuple(v) for v in C[key].values()]; lab = ["T"] + list(C[key])
    Zt = D[key].astype(np.float32).reshape(len(idx), -1); top = np.argsort(-Zt.max(1))[:20]; cell = Zt[top].argmax(1)
    I = [np.asarray(X[idx[i]]) for i in top]; allim = np.stack([hsv_edit(im, d, s) for d, s in steps for im in I])
    Z = maps(allim, tg); curves = {}
    for L, t in zip(lab, tg):
        z = Z[t]; f = z.shape[1] // 14
        if f > 1: z = z.reshape(len(z), 14, f, 14, f).max((2, 4))
        z = z.reshape(len(steps), 20, -1)[:, np.arange(20), cell]; base = z[0]
        sc = np.abs(base).mean() + 1e-6; curves[L] = np.median((z - base) / sc, 1) + 1  # 1 = unchanged
    print(f"\n{key}: curve = median change at the target's peak cell (1 = as orig); steps hue 0..330, sat .5, sat 0")
    for L in lab:
        print(f"  {L:3s} " + " ".join(f"{v:+.2f}" for v in curves[L]) + (f"   r {np.corrcoef(curves[L], curves['T'])[0,1]:+.2f}" if L != "T" else ""))
