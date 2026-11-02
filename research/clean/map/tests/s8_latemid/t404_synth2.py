"""13_404 second prediction (after textures failed): it needs a warm surface WITH a cool or dark shape on it
(blue racket on skin, dark handle on wicker). Warm field + blue/dark bar/blob > 30% of max; swapped colors < 10%."""
import sys, os, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "s6"))
from judge_lib import maps
C = json.load(open(os.path.join(H, "..", "..", "s8", "candidates.json")))["13_404"]; cs = ["C2", "C8", "C9", "C1"]
tg = [(13, 404)] + [tuple(C[c]) for c in cs]; MX = {t: json.load(open(os.path.join(H, "..", "..", "deep", f"b{t[0]}_n{t[1]}.json")))["max"] for t in tg}
rng = np.random.default_rng(0); yy, xx = np.mgrid[:224, :224]
W = {"tan": [200, 150, 105], "brown": [150, 85, 45], "skin": [225, 170, 140], "gray": [140, 140, 140], "blue": [70, 110, 190], "green": [80, 140, 50]}
K = {"blue": [40, 90, 200], "dark": [25, 25, 30], "white": [240, 240, 240], "orange": [230, 130, 30], "tan": [200, 150, 105]}
def field(c, tex=0): return np.clip(np.array(c)[None, None] + rng.normal(0, tex, (224, 224, 1)), 0, 255)
bar = (np.abs(yy - 112) < 14) & (np.abs(xx - 112) < 70); blob = (yy - 112) ** 2 + (xx - 112) ** 2 < 40 ** 2
S = {}
for fn, fc in W.items():
    for kn, kc in K.items():
        if kn == fn: continue
        for sn, m in (("bar", bar), ("blob", blob)):
            a = field(fc, 12); a[m] = kc; S[f"{fn} + {kn} {sn}"] = a.astype(np.uint8)
keys = list(S); Z = maps(np.stack([S[k] for k in keys]), tg); rows = []
for i, k in enumerate(keys): rows.append((k, [Z[t][i].max() / MX[t] for t in tg]))
rows.sort(key=lambda r: -r[1][0])
print("stimulus".ljust(24) + "".join(f"{'T' if j == 0 else cs[j-1]:>8s}" for j in range(len(tg))))
for k, r in rows[:14] + [("...", [0] * len(tg))] + rows[-8:]: print(k.ljust(24) + "".join(f"{v:+8.2f}" for v in r))
