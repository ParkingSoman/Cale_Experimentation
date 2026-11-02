"""b9 n851 P3b: bar tilt sweep (deg off vertical, + = top leans right), flat and noisy ground, dark and light bar."""
import sys, os, json, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_lib import maps
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
C = json.load(open(os.path.join(D, "s6", "candidates.json")))["9_851"]; T = (9, 851)
tg = [T] + [tuple(v) for v in C.values()]; MX = {t: json.load(open(os.path.join(D, "deep", f"b{t[0]}_n{t[1]}.json")))["max"] for t in tg}
rng = np.random.default_rng(0); NOISE = np.clip(150 + rng.normal(0, 30, (224, 224, 1)).repeat(3, -1), 0, 255).astype(np.uint8)


def bar(a, w=30, fg=40, ground="flat", L=90):
    im = Image.fromarray(NOISE.copy()) if ground == "noise" else Image.new("RGB", (224, 224), (200,) * 3); d = ImageDraw.Draw(im)
    r = np.deg2rad(a); c, s = np.cos(r), np.sin(r); pts = [(-w / 2, -L), (w / 2, -L), (w / 2, L), (-w / 2, L)]
    d.polygon([(112 + x * c - y * s, 112 + x * s + y * c) for x, y in pts], fill=(fg,) * 3); return np.array(im)


S = {}
for g in ("flat", "noise"):
    for a in (-45, -30, -20, -10, 0, 10, 20, 30, 45):
        S[f"{g}_dark_tilt{a:+d}"] = bar(a, ground=g)
    for a in (-20, 20): S[f"{g}_light_tilt{a:+d}"] = bar(a, fg=245, ground=g)
for w in (8, 16, 60): S[f"flat_dark_tilt+20_w{w}"] = bar(20, w=w)
S["noise_only"] = NOISE
keys = list(S); Z = maps(np.stack([S[k] for k in keys]), tg); res = {}
print("stimulus".ljust(24) + "   T851  " + " ".join(f"{c:>6s}" for c in C))
for i, k in enumerate(keys):
    row = [float(Z[t][i][2:-2, 2:-2].max() / MX[t]) if Z[t].shape[1] > 20 or True else 0 for t in tg]
    res[k] = row; print(k.ljust(24) + f"{row[0]:+6.2f}  " + " ".join(f"{v:+6.2f}" for v in row[1:]))
json.dump({"cands": list(C), "res": res}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "t851_tilt.json"), "w"), indent=1)
