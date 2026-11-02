"""Locality test for a target: cut a square (s px) around each top-20 peak and paste it (a) at the centre of a gray
image, (b) at the same place on a gray image, (c) keep only that square in place with the rest blurred (r=8).
Also the orig. Median image max / neuron max for target and listed candidates. Usage: python t_transplant.py KEY s C2,C8"""
import sys, os, json, numpy as np
from PIL import Image, ImageFilter
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "s6")); sys.path.insert(0, os.path.join(H, "..", "..", "..", "vision2"))
from judge_lib import maps
from convnext import load_images
key, s = sys.argv[1], int(sys.argv[2]); cs = sys.argv[3].split(",") if len(sys.argv) > 3 else []
C = json.load(open(os.path.join(H, "..", "..", "s8", "candidates.json")))[key]; tg = [tuple(map(int, key.split("_")))] + [tuple(C[c]) for c in cs]
MX = {t: json.load(open(os.path.join(H, "..", "..", "deep", f"b{t[0]}_n{t[1]}.json")))["max"] for t in tg}
D = np.load(os.path.join(H, "out", "scan.npz")); X, _ = load_images(); idx = D["idx"]
Zt = D[key].astype(np.float32).reshape(len(idx), -1); top = np.argsort(-Zt.max(1))[:20]; cell = Zt[top].argmax(1)
E = {"orig": [], "square at centre on gray": [], "square in place on gray": [], "square in place, rest blurred": [], "square in place, rest gray-mean": []}
for i, c in zip(top, cell):
    im = np.asarray(X[idx[i]]); r, q = divmod(int(c), 14); cy, cx = r * 16 + 8, q * 16 + 8
    t = int(np.clip(cy - s // 2, 0, 224 - s)); l = int(np.clip(cx - s // 2, 0, 224 - s)); sq = im[t:t + s, l:l + s]
    g = np.full_like(im, 128); a = g.copy(); a[112 - s // 2:112 - s // 2 + s, 112 - s // 2:112 - s // 2 + s] = sq; b = g.copy(); b[t:t + s, l:l + s] = sq
    bl = np.asarray(Image.fromarray(im).filter(ImageFilter.GaussianBlur(8))).copy(); bl[t:t + s, l:l + s] = sq
    gm = np.tile(im.reshape(-1, 3).mean(0).astype(np.uint8), (224, 224, 1)); gm[t:t + s, l:l + s] = sq
    for k, v in zip(E, (im, a, b, bl, gm)): E[k].append(v)
Z = maps(np.concatenate([np.stack(v) for v in E.values()]), tg); o = 0
print(f"{key}, square {s}px".ljust(34) + "".join(f"{'T' if j == 0 else cs[j-1]:>8s}" for j in range(len(tg))))
for k in E:
    print(k.ljust(34) + "".join(f"{np.median(Z[t][o:o + 20].reshape(20, -1).max(1)) / MX[t]:+8.2f}" for t in tg)); o += 20
