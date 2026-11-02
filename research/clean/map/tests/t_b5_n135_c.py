"""b5 n135 family: segment length and angle tuning of its inputs and readers (dark lines, width 3)."""
import json, numpy as np
from early_lib import *
T = [(5, 135), (4, 212), (4, 27), (3, 315), (2, 173), (4, 320), (3, 660), (8, 106), (7, 60), (6, 808)]
stim, lab = [], []
for a in (0, 10, 20, 30, 45, 90, 160):
    for L in (16, 32, 64, 300):
        stim.append(line_img(a, 3, 0, 200, length=L)); lab.append((a, L))
from PIL import Image, ImageDraw
def circle(r, width=3):
    im = Image.new("RGB", (224, 224), (200,) * 3); ImageDraw.Draw(im).ellipse([112 - r, 112 - r, 112 + r, 112 + r], outline=(0,) * 3, width=width)
    return np.array(im)
stim.append(circle(24)); lab.append(("ring24", 0)); stim.append(circle(48)); lab.append(("ring48", 0))
Z = multi(T, np.stack(stim)); out = {}
for t in T:
    r = stat(Z[t]); out[f"{t[0]},{t[1]}"] = r.tolist()
    print(f"\n{t} raw; rows angle, cols length 16/32/64/inf; then ring24 ring48")
    for k, a in enumerate((0, 10, 20, 30, 45, 90, 160)): print(f"  {a:3d}", " ".join(f"{x:6.2f}" for x in r[k * 4:(k + 1) * 4]))
    print("  rings", " ".join(f"{x:6.2f}" for x in r[-2:]))
json.dump({"labels": lab, "resp": out}, open(__file__.replace(".py", ".json"), "w"))
