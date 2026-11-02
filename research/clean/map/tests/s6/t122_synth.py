"""b12 n122 test P4 + same stimuli on candidates. Reports central max/neuron max, and for 'object' stimuli
the target's value on the object (centre cell) and far from it (corner cells)."""
import sys, os, json, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_lib import maps
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
C = json.load(open(os.path.join(D, "s6", "candidates.json")))["12_122"]; T = (12, 122)
tg = [T] + [tuple(v) for v in C.values()]; MX = {t: json.load(open(os.path.join(D, "deep", f"b{t[0]}_n{t[1]}.json")))["max"] for t in tg}
rng = np.random.default_rng(0)
def noise(sd=40, blur=0):
    a = rng.normal(128, sd, (224, 224, 3)); return np.clip(a, 0, 255).astype(np.uint8)
S = {}
for g in (40, 128, 200): S[f"uniform_{g}"] = np.full((224, 224, 3), g, np.uint8)
S["uniform_skyblue"] = np.tile(np.array([150, 190, 230], np.uint8), (224, 224, 1))
S["noise_full_sd40"] = noise(); S["noise_full_sd15"] = noise(15)
for r in (15, 30, 50):
    im = np.full((224, 224, 3), 160, np.uint8); yy, xx = np.mgrid[:224, :224]; m = (yy - 112) ** 2 + (xx - 112) ** 2 < r * r; im[m] = noise()[m]; S[f"obj_r{r}_on_plain"] = im
for r in (30, 60, 80):
    im = noise(); yy, xx = np.mgrid[:224, :224]; m = (yy - 112) ** 2 + (xx - 112) ** 2 < r * r; im[m] = 160; S[f"empty_disc_r{r}_in_noise"] = im
im = np.full((224, 224, 3), 160, np.uint8); im[:, 112:] = noise()[:, 112:]; S["half_plain_half_noise"] = im
keys = list(S); Z = maps(np.stack([S[k] for k in keys]), tg); res = {}
print("stimulus".ljust(24) + "  T122  centre  corner  " + " ".join(f"{c:>6s}" for c in C))
for i, k in enumerate(keys):
    row = [float(Z[t][i][1:-1, 1:-1].max() / MX[t]) for t in tg]; z = Z[T][i] / MX[T]
    res[k] = row + [float(z[6:8, 6:8].mean()), float(z[[1, 1, 12, 12], [1, 12, 1, 12]].mean())]
    print(k.ljust(24) + f"{row[0]:+6.2f}  {res[k][-2]:+6.2f}  {res[k][-1]:+6.2f}  " + " ".join(f"{v:+6.2f}" for v in row[1:]))
print("\nT122 map, half_plain_half_noise (row 7):", np.round(Z[T][keys.index('half_plain_half_noise')][7] / MX[T], 2))
json.dump({"cands": list(C), "res": res}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "t122_synth.json"), "w"), indent=1)
