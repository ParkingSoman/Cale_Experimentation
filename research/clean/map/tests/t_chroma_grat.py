"""Luminance vs chromatic gratings/noise for b3 n623, b4 n499 and their inputs.
Conditions built in YCbCr (Y mean 128): lum = Y +-50; by = Cb +-40 (isoluminant blue-yellow); rg = Cr +-40;
yb+ = bright is yellow, dark is blue (Y +-50 with Cb -+40); yb- = bright is blue; tintY/tintB = luminance grating on a
yellowish (Cb -25) or bluish (Cb +25) mean. Vertical square gratings, periods 4..48 px, and 2 px-grain noise."""
import json, numpy as np
from PIL import Image
from early_lib import *
from t_chroma import rgb

T = [(3, 623), (4, 499), (1, 365), (0, 346), (0, 364), (1, 136), (4, 401), (4, 395), (5, 375), (5, 302), (5, 189)]
NAT = {(3, 623): 3.01, (4, 499): 4.51, (1, 365): 15.52, (0, 346): 18.75, (0, 364): 14.6}
COND = {"lum": (50, 0, 0, 0), "by": (0, 40, 0, 0), "rg": (0, 0, 40, 0), "yb+": (50, -40, 0, 0), "yb-": (50, 40, 0, 0),
        "rg+": (50, 0, 40, 0), "tintY": (50, 0, 0, -25), "tintB": (50, 0, 0, 25), "flatY": (0, 0, 0, -25), "flatB": (0, 0, 0, 25)}
PER = (4, 8, 16, 32, 48); rng = np.random.default_rng(1)
nz = np.array(Image.fromarray(rng.choice([0, 255], (112, 112)).astype(np.uint8)).resize((224, 224), Image.NEAREST)) / 127.5 - 1
stim, lab = [], []
for c, (ay, ab, ar, mb) in COND.items():
    for p in PER + ("noise",):
        s = nz if p == "noise" else (grating(90, p, .5, 0, 255).astype(np.float32)[..., 0] / 127.5 - 1)
        stim.append(rgb(np.stack([128 + ay * s, mb + ab * s, ar * s], -1))); lab.append((c, p))
Z = multi(T, np.stack(stim)); R = {f"{b},{n}": stat(Z[(b, n)]).tolist() for b, n in T}
json.dump({"labels": [list(map(str, l)) for l in lab], "resp": R}, open(__file__.replace(".py", ".json"), "w"))
for b, n in T:
    r = np.array(R[f"{b},{n}"]); nat = NAT.get((b, n)); r = r / nat if nat else r
    print(f"\n{b},{n} {'(fraction of natural max)' if nat else '(raw)'}  cols = period " + " ".join(map(str, PER)) + " noise")
    for k, c in enumerate(COND):
        print(f"  {c:6s}", " ".join(f"{x:6.2f}" for x in r[k * 6:(k + 1) * 6]))
