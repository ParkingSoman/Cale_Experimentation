"""b9 n851 test P3 + the same stimuli on the 9 candidates (value = central max / neuron max)."""
import sys, os, json, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from judge_lib import maps
from early_lib import grating
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
key = "9_851"; C = json.load(open(os.path.join(D, "s6", "candidates.json")))[key]; T = (9, 851)
tg = [T] + [tuple(v) for v in C.values()]; MX = {t: json.load(open(os.path.join(D, "deep", f"b{t[0]}_n{t[1]}.json")))["max"] for t in tg}


def cv(bg=200): im = Image.new("RGB", (224, 224), (bg,) * 3); return im, ImageDraw.Draw(im)


S = {}
for w in (12, 30, 60):
    im, d = cv(); d.rectangle([112 - w // 2, 20, 112 + w // 2, 204], fill=(40,) * 3); S[f"vbar_dark_w{w}"] = np.array(im)
    im, d = cv(); d.rectangle([20, 112 - w // 2, 204, 112 + w // 2], fill=(40,) * 3); S[f"hbar_dark_w{w}"] = np.array(im)
im, d = cv(60); d.rectangle([97, 20, 127, 204], fill=(220,) * 3); S["vbar_light_on_dark_w30"] = np.array(im)
im, d = cv(); d.rectangle([97, 0, 127, 224], fill=(40,) * 3); S["vbar_full_height_w30"] = np.array(im)
im, d = cv(); d.rectangle([0, 0, 112, 224], fill=(40,) * 3); S["vedge_halfplane"] = np.array(im)
im, d = cv(); d.rectangle([0, 0, 224, 112], fill=(40,) * 3); S["hedge_halfplane"] = np.array(im)
im, d = cv()
d.rectangle([60, 50, 170, 120], fill=(40,) * 3)
for x in (65, 85, 145, 160): d.rectangle([x, 120, x + 12, 200], fill=(40,) * 3)
S["silhouette_4legs"] = np.array(im)
im, d = cv(); d.rectangle([60, 50, 170, 120], fill=(40,) * 3); S["silhouette_body_only"] = np.array(im)
im, d = cv()
for x in (65, 85, 145, 160): d.rectangle([x, 60, x + 12, 200], fill=(40,) * 3)
S["four_bars_w12"] = np.array(im)
for a in (0, 30, 60, 90):
    im, d = cv(); r = np.deg2rad(a); c, s = np.cos(r), np.sin(r); pts = [(-15, -90), (15, -90), (15, 90), (-15, 90)]
    d.polygon([(112 + x * c - y * s, 112 + x * s + y * c) for x, y in pts], fill=(40,) * 3); S[f"bar_rot{a}"] = np.array(im)
S["grating_vert_p16"] = grating(90, 16, lo=40, hi=200); S["grating_vert_p48"] = grating(90, 48, lo=40, hi=200)
S["flat"] = np.full((224, 224, 3), 200, np.uint8)
keys = list(S); Z = maps(np.stack([S[k] for k in keys]), tg); res = {}
print("stimulus".ljust(24) + "   T851 peak(r,c) " + " ".join(f"{c:>6s}" for c in C))
for i, k in enumerate(keys):
    row = [float(Z[t][i][2:-2, 2:-2].max() / MX[t]) for t in tg]; z = Z[T][i]; r, c = np.unravel_index(z[2:-2, 2:-2].argmax(), (10, 10))
    res[k] = row; print(k.ljust(24) + f"{row[0]:+6.2f} ({r+2:2d},{c+2:2d})  " + " ".join(f"{v:+6.2f}" for v in row[1:]))
print("\nT851 map rows 3-10 for vbar_dark_w30:"); print(np.round(Z[T][keys.index('vbar_dark_w30')][3:11] / MX[T], 1))
json.dump({"cands": list(C), "res": res}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "t851_synth.json"), "w"), indent=1)
