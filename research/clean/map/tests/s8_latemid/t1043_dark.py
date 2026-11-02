"""14_1043: on-cells are dark (median brightness 59 vs 110). Is darkness causal? Top-20 and 200 random images:
orig / x0.5 / x0.3 (darker) / brighten (gamma .5) / invert. Flat fields at 20, 60, 128. Median image max / max."""
import sys, os, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "s6")); sys.path.insert(0, os.path.join(H, "..", "..", "..", "vision2"))
from judge_lib import maps
from convnext import load_images
C = json.load(open(os.path.join(H, "..", "..", "s8", "candidates.json")))["14_1043"]; cs = ["C8", "C9", "C7", "C5"]
tg = [(14, 1043)] + [tuple(C[c]) for c in cs]; MX = {t: json.load(open(os.path.join(H, "..", "..", "deep", f"b{t[0]}_n{t[1]}.json")))["max"] for t in tg}
D = np.load(os.path.join(H, "out", "scan.npz")); X, _ = load_images(); idx = D["idx"]
top = np.argsort(-D["14_1043"].astype(np.float32).reshape(len(idx), -1).max(1))[:20]; rnd = np.random.default_rng(2).choice(len(idx), 200, replace=False)
E = {}
for nm, sel in (("top20", top), ("rand200", rnd)):
    I = np.stack([np.asarray(X[idx[i]]) for i in sel]).astype(np.float32)
    E[f"{nm} orig"] = I; E[f"{nm} x0.5"] = I * .5; E[f"{nm} x0.3"] = I * .3; E[f"{nm} brighten g.5"] = 255 * (I / 255) ** .5; E[f"{nm} invert"] = 255 - I
for v in (20, 60, 128): E[f"flat {v}"] = np.full((1, 224, 224, 3), v, np.float32)
Z = maps(np.concatenate([np.clip(a, 0, 255).astype(np.uint8) for a in E.values()]), tg); o = 0
print("edit".ljust(22) + "".join(f"{'T' if j == 0 else cs[j-1]:>8s}" for j in range(len(tg))))
for k, a in E.items():
    n = len(a); print(k.ljust(22) + "".join(f"{np.median(Z[t][o:o + n].reshape(n, -1).max(1)) / MX[t]:+8.2f}" for t in tg)); o += n
