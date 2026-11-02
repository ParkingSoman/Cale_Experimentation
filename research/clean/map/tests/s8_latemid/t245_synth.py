"""12_245 prediction test: pale blue drives it only next to a boundary (horizon to a darker or warmer region, or the
frame edge); full uniform blue stays near 0. Image max / neuron max for target and blue-dial candidates."""
import sys, os, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "s6"))
from judge_lib import maps
C = json.load(open(os.path.join(H, "..", "..", "s8", "candidates.json")))["12_245"]; cs = ["C2", "C6", "C9", "C1", "C7", "C8"]
tg = [(12, 245)] + [tuple(C[c]) for c in cs]; MX = {t: json.load(open(os.path.join(H, "..", "..", "deep", f"b{t[0]}_n{t[1]}.json")))["max"] for t in tg}
rng = np.random.default_rng(0); sky = np.array([170, 195, 225]); dark = np.array([40, 45, 50]); orange = np.array([200, 120, 40]); green = np.array([60, 110, 40])
def full(c): return np.tile(np.array(c, np.uint8), (224, 224, 1))
def split(top, bot, y=112): a = full(bot); a[:y] = top; return a
def tex(c, sd=35): return np.clip(np.array(c)[None, None] + rng.normal(0, sd, (224, 224, 3)), 0, 255).astype(np.uint8)
S = {"full pale blue": full(sky), "full gray": full([128] * 3), "full dark": full(dark),
     "sky over dark (horizon)": split(sky, dark), "sky over textured dark": np.where(np.arange(224)[:, None, None] < 112, full(sky), tex(dark)),
     "sky over green texture": np.where(np.arange(224)[:, None, None] < 112, full(sky), tex(green)),
     "sky over orange": split(sky, orange), "gray over dark": split([170, 170, 170], dark), "orange over dark": split(orange, dark),
     "dark disc in sky": None, "sky disc in dark": None, "textured blue (noise)": tex(sky)}
yy, xx = np.mgrid[:224, :224]; m = (yy - 112) ** 2 + (xx - 112) ** 2 < 60 ** 2
a = full(sky); a[m] = dark; S["dark disc in sky"] = a; a = full(dark); a[m] = sky; S["sky disc in dark"] = a
keys = list(S); Z = maps(np.stack([S[k] for k in keys]), tg)
print("stimulus".ljust(26) + "".join(f"{'T' if j == 0 else cs[j-1]:>8s}" for j in range(len(tg))) + "   T peak (row,col)")
for i, k in enumerate(keys):
    row = [Z[t][i].max() / MX[t] for t in tg]; a = Z[tg[0]][i].argmax()
    print(k.ljust(26) + "".join(f"{v:+8.2f}" for v in row) + f"   {divmod(int(a), 14)}")
