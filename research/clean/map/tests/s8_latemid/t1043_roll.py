"""14_1043 follow-up: column profiles (median over top-20 images of column max / neuron max) for orig, horizontal
roll by 112 px (old left/right edges meet at the centre seam, cols 6-7), reflect-shift 48 px, gray-shift 48 px,
and a 'gray frame' (outer 32 px replaced by gray, content in place). Usage: python t1043_roll.py"""
import sys, os, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "s6")); sys.path.insert(0, os.path.join(H, "..", "..", "..", "vision2"))
from judge_lib import maps
from convnext import load_images
key = "14_1043"; C = json.load(open(os.path.join(H, "..", "..", "s8", "candidates.json")))[key]
cs = ["C8", "C9", "C7", "C5", "C1"]; tg = [(14, 1043)] + [tuple(C[c]) for c in cs]
MX = {t: json.load(open(os.path.join(H, "..", "..", "deep", f"b{t[0]}_n{t[1]}.json")))["max"] for t in tg}
D = np.load(os.path.join(H, "out", "scan.npz")); X, _ = load_images(); idx = D["idx"]
top = np.argsort(-D[key].astype(np.float32).reshape(len(idx), -1).max(1))[:20]; I = np.stack([np.asarray(X[idx[i]]) for i in top])
refl = np.pad(I, ((0, 0), (0, 0), (48, 0), (0, 0)), mode="reflect")[:, :, :224]
gsh = np.full_like(I, 128); gsh[:, :, 48:] = I[:, :, :176]
frame = I.copy(); frame[:, :32] = 128; frame[:, -32:] = 128; frame[:, :, :32] = 128; frame[:, :, -32:] = 128
E = {"orig": I, "roll112": np.roll(I, 112, axis=2), "reflect48": refl, "grayshift48": gsh, "grayframe32": frame}
Z = maps(np.concatenate(list(E.values())), tg); o = 0
for name, arr in E.items():
    n = len(arr); print(f"\n{name}  (cols 0..13, median column max / neuron max; last = image max)")
    for j, t in enumerate(tg):
        z = Z[t][o:o + n] / MX[t]; prof = np.median(z.max(1), 0)
        print(f"  {'T ' if j == 0 else cs[j-1]:3s}" + " ".join(f"{v:+.2f}" for v in prof) + f"  | {np.median(z.reshape(n, -1).max(1)):+.2f}")
    o += n
