"""12_575 follow-up: dark ellipse (ry = 4/3 rx) on plain light ground; size sweep at centre and position sweep at
ry=20. Value at the blob's cell(s) (max over a 3x3 window around it) / neuron max, for T, C6, C8, C9."""
import sys, os, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "s6"))
from judge_lib import maps
C = json.load(open(os.path.join(H, "..", "..", "s8", "candidates.json")))["12_575"]; cs = ["C6", "C8", "C9", "C2"]
tg = [(12, 575)] + [tuple(C[c]) for c in cs]; J = {t: json.load(open(os.path.join(H, "..", "..", "deep", f"b{t[0]}_n{t[1]}.json"))) for t in tg}
yy, xx = np.mgrid[:224, :224]; S = {}
def img(cy, cx, ry):
    a = np.full((224, 224, 3), 200, np.uint8); a[((yy - cy) / ry) ** 2 + ((xx - cx) / (ry * .75)) ** 2 < 1] = 40; return a
for ry in (6, 10, 14, 20, 28, 40): S[f"centre ry{ry}"] = (img(112, 112, ry), 112, 112)
for nm, (cy, cx) in {"row1": (24, 112), "row4": (72, 112), "row10": (168, 112), "row12": (200, 112), "col1": (112, 24), "col12": (112, 200)}.items():
    S[f"ry20 {nm}"] = (img(cy, cx, 20), cy, cx)
keys = list(S); Z = maps(np.stack([S[k][0] for k in keys]), tg)
print("stimulus".ljust(16) + "".join(f"{'T' if j == 0 else cs[j-1]:>8s}" for j in range(len(tg))))
for i, k in enumerate(keys):
    r, c = S[k][1] // 16, S[k][2] // 16; s = k.ljust(16)
    for t in tg:
        z = Z[t][i]; f = z.shape[0] // 14; w = z[max(0, (r - 1) * f):(r + 2) * f, max(0, (c - 1) * f):(c + 2) * f]; s += f"{w.max() / J[t]['max']:+8.2f}"
    print(s)
