"""13_404 prediction: muted orange-brown TEXTURE (noise, stripes, brick grid) drives it > 50% of max; the same texture
in blue or gray < 10%; flat orange stays below 0. Image max / neuron max for target and candidates."""
import sys, os, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "s6"))
from judge_lib import maps
C = json.load(open(os.path.join(H, "..", "..", "s8", "candidates.json")))["13_404"]; cs = ["C2", "C8", "C9", "C1", "C4"]
tg = [(13, 404)] + [tuple(C[c]) for c in cs]; MX = {t: json.load(open(os.path.join(H, "..", "..", "deep", f"b{t[0]}_n{t[1]}.json")))["max"] for t in tg}
rng = np.random.default_rng(0); yy, xx = np.mgrid[:224, :224]
COL = {"brown": [150, 85, 45], "orange": [220, 130, 40], "tan": [190, 150, 100], "yellow": [220, 200, 50], "green": [80, 140, 50], "blue": [60, 100, 180], "gray": [128, 128, 128], "red": [180, 40, 40]}
def flat(c): return np.tile(np.array(c, np.uint8), (224, 224, 1))
def noise(c, sd=40): return np.clip(np.array(c)[None, None] + rng.normal(0, sd, (224, 224, 1)), 0, 255).astype(np.uint8)
def stripes(c, p=10): return np.clip(np.array(c)[None, None] * (0.7 + 0.5 * (np.sin(2 * np.pi * yy / p) > 0))[..., None], 0, 255).astype(np.uint8)
def brick(c): g = ((yy % 16 < 2) | (((xx + 16 * ((yy // 16) % 2)) % 32) < 2)); a = flat(c).astype(float); a[g] = 220; return a.astype(np.uint8)
S = {}
for n, c in COL.items():
    S[f"{n} flat"] = flat(c); S[f"{n} noise"] = noise(c); S[f"{n} stripes"] = stripes(c); S[f"{n} brick"] = brick(c)
keys = list(S); Z = maps(np.stack([S[k] for k in keys]), tg)
print("stimulus".ljust(18) + "".join(f"{'T' if j == 0 else cs[j-1]:>8s}" for j in range(len(tg))))
for i, k in enumerate(keys):
    row = []
    for t in tg:
        z = Z[t][i]; f = z.shape[0] // 14; row.append(z[2 * f:12 * f, 2 * f:12 * f].max() / MX[t])
    print(k.ljust(18) + "".join(f"{v:+8.2f}" for v in row))
