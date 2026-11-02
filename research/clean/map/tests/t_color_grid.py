"""Color grid for b3 n623 and b4 n499 (+ their inputs, candidates and readers): uniform and noisy patches over
hue (24) x saturation (0.15, 0.4, 0.8) x value (0.35, 0.6, 0.85), plus grays."""
import json, colorsys, numpy as np
from early_lib import *

T = [(3, 623), (0, 346), (0, 364), (2, 61), (0, 200), (1, 309), (4, 401), (4, 467), (4, 395),
     (4, 499), (1, 365), (1, 136), (0, 32), (5, 375), (5, 302), (5, 189)]
H = np.arange(24) * 15; S = (0.15, 0.4, 0.8); V = (0.35, 0.6, 0.85); rng = np.random.default_rng(0)
noise = rng.normal(0, 22, (224, 224, 1))
cols, lab = [], []
for h in H:
    for s in S:
        for v in V:
            cols.append(np.array(colorsys.hsv_to_rgb(h / 360, s, v)) * 255); lab.append((int(h), s, v))
for g in (0.2, 0.35, 0.6, 0.85): cols.append(np.array([g, g, g]) * 255); lab.append(("gray", 0, g))
flat = np.stack([patch(c.round().astype(np.uint8)) for c in cols])
noisy = np.stack([(c[None, None] + noise).clip(0, 255).astype(np.uint8) for c in cols])
Zf = multi(T, flat); Zn = multi(T, noisy)
R = {f"{b},{n}": {"flat": stat(Zf[(b, n)]).tolist(), "noisy": stat(Zn[(b, n)]).tolist()} for b, n in T}
json.dump({"labels": [list(map(str, l)) for l in lab], "resp": R}, open(__file__.replace(".py", ".json"), "w"))


def table(key, nat, kind):
    r = np.array(R[key][kind]) / nat
    print(f"\n{key} {kind} (fraction of natural max {nat}); rows hue, cols s/v = " + " ".join(f"{s}/{v}" for s in S for v in V))
    for i, h in enumerate(H):
        print(f"  {h:3d}", " ".join(f"{x:5.2f}" for x in r[i * 9:(i + 1) * 9]))
    print("  gray", " ".join(f"{x:5.2f}" for x in r[-4:]))


for key, nat in (("3,623", 3.01), ("4,499", 4.51)):
    table(key, nat, "flat"); table(key, nat, "noisy")
# correlation of each other neuron's hue curve with the target (noisy, all cells)
for tgt, others in (("3,623", ["0,346", "0,364", "2,61", "0,200", "1,309", "4,401", "4,467", "4,395"]),
                    ("4,499", ["1,365", "1,136", "0,32", "1,309", "5,375", "5,302", "5,189"])):
    a = np.array(R[tgt]["noisy"])
    print("\ncorr with", tgt, "over the noisy grid:")
    for o in others:
        bvec = np.array(R[o]["noisy"]); best = int(np.argmax(bvec)); worst = int(np.argmin(bvec))
        print(f"  {o:7s} r={np.corrcoef(a, bvec)[0,1]:+.2f}  peak {lab[best]} ({bvec[best]:.2f})  trough {lab[worst]} ({bvec[worst]:.2f})")
