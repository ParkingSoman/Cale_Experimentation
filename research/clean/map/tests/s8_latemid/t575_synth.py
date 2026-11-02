"""12_575 prediction: a compact shape on a plain ground drives it at the shape's middle (> 50% of max, centred);
the same shape at the bottom edge < 20%; a textured ground band at the bottom goes negative. Also off-pole
readings: the min of each neuron, to test opponents C5/C7. Image max / neuron max (and min / |neuron min|)."""
import sys, os, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "s6"))
from judge_lib import maps
C = json.load(open(os.path.join(H, "..", "..", "s8", "candidates.json")))["12_575"]; cs = ["C6", "C8", "C9", "C2", "C5", "C7"]
tg = [(12, 575)] + [tuple(C[c]) for c in cs]; J = {t: json.load(open(os.path.join(H, "..", "..", "deep", f"b{t[0]}_n{t[1]}.json"))) for t in tg}
rng = np.random.default_rng(0); yy, xx = np.mgrid[:224, :224]
def ground(g=200): return np.full((224, 224, 3), g, np.float32)
def tex(a, m, c=110, sd=40): a[m] = np.clip(c + rng.normal(0, sd, (m.sum(), 1)), 0, 255); return a
def ell(cy, cx, ry, rx): return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1
S = {}
for nm, (cy, cx) in {"centre": (112, 112), "upper": (50, 112), "bottom edge": (205, 112), "left edge": (112, 15)}.items():
    a = ground(); a[ell(cy, cx, 40, 30)] = 40; S[f"dark blob {nm}"] = a
a = ground(); S["textured blob centre"] = tex(a, ell(112, 112, 40, 30))
a = ground(60); a[ell(112, 112, 40, 30)] = 220; S["light blob on dark centre"] = a
a = ground(); a[ell(112, 112, 70, 55)] = 40; S["big dark blob centre"] = a
a = ground(); a[ell(112, 112, 20, 15)] = 40; S["small dark blob centre"] = a
S["plain ground"] = ground()
a = ground(); S["textured floor band bottom"] = tex(a, yy > 150)
a = ground(); S["textured floor band top"] = tex(a, yy < 74)
a = ground(); a[ell(112, 112, 40, 30)] = 40; S["blob + floor band"] = tex(a, yy > 165)
S["full texture"] = tex(ground(), yy >= 0)
keys = list(S); Z = maps(np.stack([S[k] for k in keys]).astype(np.uint8), tg)
print("stimulus".ljust(28) + "".join(f"{'T' if j == 0 else cs[j-1]:>13s}" for j in range(len(tg))) + "   T peak / T min (row,col)")
for i, k in enumerate(keys):
    s = k.ljust(28)
    for t in tg:
        z = Z[t][i]; s += f"  {z.max() / J[t]['max']:+.2f}/{z.min() / abs(J[t]['min']):+.2f}"
    z = Z[tg[0]][i]; print(s + f"   {divmod(int(z.argmax()), 14)} {divmod(int(z.argmin()), 14)}")
