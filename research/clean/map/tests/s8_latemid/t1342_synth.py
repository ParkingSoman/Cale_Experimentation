"""12_1342 prediction: a gray-scale field crowded with small shaded lobes / short fingers drives it > 50% of max;
the same field blurred (r=3) < 20%; flat or parallel straight lines go negative. Dot fields (round units) are scored
too, to separate 'many round units' (candidates C8/C5?) from 'many lobes'. Image max (inner 10x10) / neuron max."""
import sys, os, json, numpy as np
from PIL import Image, ImageDraw, ImageFilter
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "s6"))
from judge_lib import maps
C = json.load(open(os.path.join(H, "..", "..", "s8", "candidates.json")))["12_1342"]; cs = ["C8", "C5", "C9", "C1"]
tg = [(12, 1342)] + [tuple(C[c]) for c in cs]; MX = {t: json.load(open(os.path.join(H, "..", "..", "deep", f"b{t[0]}_n{t[1]}.json")))["max"] for t in tg}
rng = np.random.default_rng(0)
def canvas(bg=90): return Image.new("L", (224, 224), bg)
def fingers(n=180, L=(10, 22), w=(5, 9), shade=True, bg=60):
    im = canvas(bg); d = ImageDraw.Draw(im)
    for _ in range(n):
        x, y = rng.uniform(0, 224, 2); a = rng.uniform(0, np.pi); l = rng.uniform(*L); ww = int(rng.uniform(*w))
        x2, y2 = x + l * np.cos(a), y + l * np.sin(a); g = int(rng.uniform(150, 240))
        d.line([x, y, x2, y2], fill=g, width=ww); d.ellipse([x2 - ww / 2, y2 - ww / 2, x2 + ww / 2, y2 + ww / 2], fill=g)
        if shade: d.line([x + 1, y + 1, x2 + 1, y2 + 1], fill=max(g - 70, 0), width=max(ww // 3, 1))
    return im
def dots(n, r, bg=60):
    im = canvas(bg); d = ImageDraw.Draw(im)
    for _ in range(n):
        x, y = rng.uniform(0, 224, 2); g = int(rng.uniform(150, 240)); d.ellipse([x - r, y - r, x + r, y + r], fill=g, outline=max(g - 90, 0))
    return im
def lines(p=12):
    a = np.full((224, 224), 90, np.uint8); a[:, ::p] = 220; a[:, 1::p] = 220; return Image.fromarray(a)
S = {"flat gray": canvas(), "parallel lines": lines(), "fingers shaded": fingers(), "fingers flat": fingers(shade=False),
     "fingers sparse (60)": fingers(60), "fingers big": fingers(90, (20, 40), (10, 16)), "fingers small": fingers(400, (5, 10), (3, 5)),
     "dots r4 dense": dots(700, 4), "dots r8 dense": dots(260, 8), "dots r14": dots(90, 14), "noise sd50": Image.fromarray(np.clip(rng.normal(128, 50, (224, 224)), 0, 255).astype(np.uint8))}
S["fingers shaded blur3"] = S["fingers shaded"].filter(ImageFilter.GaussianBlur(3)); S["dots r8 blur3"] = S["dots r8 dense"].filter(ImageFilter.GaussianBlur(3))
keys = list(S); arr = np.stack([np.repeat(np.asarray(S[k])[..., None], 3, -1) for k in keys]); Z = maps(arr, tg)
print("stimulus".ljust(22) + "".join(f"{'T' if j == 0 else cs[j-1]:>8s}" for j in range(len(tg))))
for i, k in enumerate(keys):
    row = []
    for t in tg:
        z = Z[t][i]; f = z.shape[0] // 14; row.append(z[2 * f:12 * f, 2 * f:12 * f].max() / MX[t])
    print(k.ljust(22) + "".join(f"{v:+8.2f}" for v in row))
S["fingers shaded"].save(os.path.join(H, "out", "t1342_fingers.png")); S["dots r8 dense"].save(os.path.join(H, "out", "t1342_dots.png"))
