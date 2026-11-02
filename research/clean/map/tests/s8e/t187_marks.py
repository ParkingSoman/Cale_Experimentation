"""4 n187 vs candidates C9 (1,118), C2 (1,206), C7 (1,193), C1 (2,46): single marks vs clusters of marks.
Prediction (4 n187): a cluster of short dark marks (glyph-like) > 2x a single short dash; long line < 0."""
import sys, os, json, numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from early_lib import *
H = os.path.dirname(os.path.abspath(__file__)); DEEP = os.path.join(H, "..", "..", "deep")
T = [(4, 187), (1, 118), (1, 206), (1, 193), (2, 46), (2, 346)]
rng = np.random.default_rng(0)
def canvas(fn, bg=225):
    im = Image.new("RGB", (224, 224), (bg,) * 3); fn(ImageDraw.Draw(im)); return np.array(im)
def dashes(n, spread, L=8, w=2):
    def f(d):
        for _ in range(n):
            x, y = 112 + rng.uniform(-spread, spread, 2); a = rng.uniform(0, np.pi)
            d.line([(x - L / 2 * np.cos(a), y - L / 2 * np.sin(a)), (x + L / 2 * np.cos(a), y + L / 2 * np.sin(a))], fill=(20,) * 3, width=w)
    return f
S = {"single-dash": canvas(dashes(1, 0)), "single-dot4": canvas(lambda d: d.ellipse([110, 110, 114, 114], fill=(20,) * 3)),
     "cluster3": canvas(dashes(3, 8)), "cluster6": canvas(dashes(6, 12)), "cluster12": canvas(dashes(12, 16)),
     "dots-grid": canvas(lambda d: [d.ellipse([x, y, x + 3, y + 3], fill=(20,) * 3) for x in range(88, 136, 8) for y in range(88, 136, 8)]),
     "long-line": canvas(lambda d: d.line([(0, 112), (224, 112)], fill=(20,) * 3, width=2)),
     "long-line-v": canvas(lambda d: d.line([(112, 0), (112, 224)], fill=(20,) * 3, width=2))}
for sz in (10, 16, 24, 40):
    try: f = ImageFont.load_default(size=sz)
    except TypeError: f = ImageFont.load_default()
    S[f"glyph-a{sz}"] = canvas(lambda d: d.text((112, 112), "a", fill=(20,) * 3, font=f, anchor="mm"))
    S[f"word{sz}"] = canvas(lambda d: d.text((112, 112), "rainbow", fill=(20,) * 3, font=f, anchor="mm"))
    S[f"line-of-text{sz}"] = canvas(lambda d: d.multiline_text((112, 112), "the quick brown\nfox jumps over\nthe lazy dog 1980", fill=(20,) * 3, font=f, anchor="mm"))
X = np.stack(list(S.values())); Z = multi(T, X)
print(f"{'stim':18s}" + "".join(f"{str(t):>11s}" for t in T))
out = {}
for j, s in enumerate(S):
    row = []
    for t in T:
        nm = json.load(open(os.path.join(DEEP, f"b{t[0]}_n{t[1]}.json")))["max"]; k = Z[t].shape[1] // 4
        row.append(float(Z[t][j, k:-k, k:-k].max() / nm))
    out[s] = row; print(f"{s:18s}" + "".join(f"{v:+11.2f}" for v in row))
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
