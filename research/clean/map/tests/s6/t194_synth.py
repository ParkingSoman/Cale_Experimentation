"""b2 n194 synthetic test (P1) + same stimuli on the 9 candidates. Value = max over central region (z / neuron max)."""
import sys, os, json, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from judge_lib import maps
from early_lib import stat, grating
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
C = json.load(open(os.path.join(D, "s6", "candidates.json")))["2_194"]
tg = [(2, 194)] + [tuple(v) for v in C.values()]; MX = {t: json.load(open(os.path.join(D, "deep", f"b{t[0]}_n{t[1]}.json")))["max"] for t in tg}
MN = {t: json.load(open(os.path.join(D, "deep", f"b{t[0]}_n{t[1]}.json")))["min"] for t in tg}


def edge(angle, lo=60, hi=190):
    yy, xx = np.mgrid[0:224, 0:224].astype(np.float32) - 112; a = np.deg2rad(angle)
    u = -xx * np.sin(a) + yy * np.cos(a); g = np.where(u < 0, hi, lo).astype(np.float32)   # 0 = horizontal, light above
    return np.repeat(g[..., None], 3, -1).astype(np.uint8)


def hline(width, fg, bg=128, half=False, angle=0):
    im = Image.new("RGB", (224, 224), (bg,) * 3); d = ImageDraw.Draw(im); a = np.deg2rad(angle); L = 150
    dx, dy = np.cos(a) * L, -np.sin(a) * L
    p0 = (112, 112) if half else (112 - dx, 112 - dy); d.line([p0, (112 + dx, 112 + dy)], fill=(fg,) * 3, width=width); return np.array(im)


S = {}
for a in range(0, 180, 15): S[f"edge{a:03d}"] = edge(a); S[f"edge{a:03d}inv"] = edge(a, 190, 60)
for a in (0, 45, 90, 135): S[f"line_dark_w3_{a}"] = hline(3, 30, angle=a); S[f"line_light_w3_{a}"] = hline(3, 230, angle=a)
S["halfline_right_dark"] = hline(3, 30, half=True); S["halfline_left_dark"] = hline(3, 30, half=True, angle=180)
for p in (3, 4, 6, 8, 12, 16, 24):
    S[f"grat_vert_p{p}"] = grating(90, p, lo=40, hi=200); S[f"grat_horz_p{p}"] = grating(0, p, lo=40, hi=200)
S["flat_gray"] = np.full((224, 224, 3), 128, np.uint8)
keys = list(S); imgs = np.stack([S[k] for k in keys]); Z = maps(imgs, tg)
res = {}
print("stimulus".ljust(22) + "  T194  " + " ".join(f"{c:>6s}" for c in C))
for i, k in enumerate(keys):
    row = [float(stat(Z[t][i:i + 1])[0] / MX[t]) for t in tg]
    mn = float(Z[(2, 194)][i, 11:45, 11:45].min() / abs(MN[(2, 194)]))
    res[k] = row + [mn]; print(k.ljust(22) + " ".join(f"{v:+6.2f}" for v in row) + f"   T min {mn:+.2f}")
json.dump({"cands": list(C), "res": res}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "t194_synth.json"), "w"), indent=1)
