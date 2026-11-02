"""Color grid for 12_245 and 13_404 and all their candidates: uniform full images over hue (12) x saturation
(0,.3,.6,1) x value (.35,.65,.95). Score = max over the inner 10x10 cells / neuron max (no frame effect), also ring max.
Prints, per neuron: best 5 colors, worst 3, and gray-axis values. Saves out/t_color.json. Usage: python t_color.py"""
import sys, os, json, colorsys, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "s6"))
from judge_lib import maps
C = json.load(open(os.path.join(H, "..", "..", "s8", "candidates.json")))
tg = [(12, 245)] + [tuple(v) for v in C["12_245"].values()] + [(13, 404)] + [tuple(v) for v in C["13_404"].values()]
lab = ["T245"] + [f"245{c}" for c in C["12_245"]] + ["T404"] + [f"404{c}" for c in C["13_404"]]
MX = {t: json.load(open(os.path.join(H, "..", "..", "deep", f"b{t[0]}_n{t[1]}.json")))["max"] for t in tg}
cols, names = [], []
for h in range(12):
    for s in (0.3, 0.6, 1.0):
        for v in (0.35, 0.65, 0.95):
            cols.append(colorsys.hsv_to_rgb(h / 12, s, v)); names.append(f"h{h*30:03d} s{s:.1f} v{v:.2f}")
for v in (0.1, 0.35, 0.65, 0.95): cols.append((v, v, v)); names.append(f"gray v{v:.2f}")
imgs = np.stack([np.tile((np.array(c) * 255).astype(np.uint8), (224, 224, 1)) for c in cols])
Z = maps(imgs, list(dict.fromkeys(tg))); res = {}
for L, t in zip(lab, tg):
    z = Z[t]; f = z.shape[1] // 14; inner = z[:, 2 * f:12 * f, 2 * f:12 * f].reshape(len(z), -1).max(1) / MX[t]
    res[L] = dict(zip(names, map(float, inner))); o = np.argsort(-inner)
    print(f"{L:8s} {t}  best: " + ", ".join(f"{names[i]} {inner[i]:+.2f}" for i in o[:5]) + "  | worst: " + ", ".join(f"{names[i]} {inner[i]:+.2f}" for i in o[-3:]) + "  | grays " + " ".join(f"{inner[i]:+.2f}" for i in range(len(names) - 4, len(names))))
json.dump(res, open(os.path.join(H, "out", "t_color.json"), "w"), indent=0)
T = np.array([res["T245"][n] for n in names]); U = np.array([res["T404"][n] for n in names])
print("\ncorrelation of color tuning with the target (across the 112 colors):")
for L in lab:
    a = np.array([res[L][n] for n in names]); print(f"  {L:8s} r(T245) {np.corrcoef(a, T)[0,1]:+.2f}  r(T404) {np.corrcoef(a, U)[0,1]:+.2f}")
