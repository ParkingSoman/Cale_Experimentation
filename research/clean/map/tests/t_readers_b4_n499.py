"""Do the readers / inputs of b4 n499 share its 'blue above, yellow below' tuning? Colour boundary rotated
through 360 deg (with and without luminance noise), raw z."""
import json, numpy as np
from PIL import Image
from early_lib import *
from t_chroma import rgb
T = [(4, 499), (5, 375), (5, 302), (5, 189), (1, 365), (0, 32), (3, 16), (3, 169), (1, 309)]
yy, xx = np.mgrid[0:224, 0:224].astype(np.float32) - 111.5; rs = np.random.default_rng(3)
s = np.array(Image.fromarray(rs.choice([0, 255], (112, 112)).astype(np.uint8)).resize((224, 224), Image.NEAREST)) / 127.5 - 1
stim, lab = [], []
for amp in (0, 40):
    for ang in range(0, 360, 45):
        t = np.deg2rad(ang); side = (xx * np.cos(t) - yy * np.sin(t)) > 0
        c = np.zeros((224, 224, 3), np.float32); c[..., 0] = 128 + amp * s; c[..., 1] = np.where(side, 30, -30); stim.append(rgb(c)); lab.append((amp, ang))
Z = multi(T, np.stack(stim)); out = {}
print("blue side direction: 90 = blue above. cols = 0 45 90 135 180 225 270 315; first row no noise, second with noise (raw z)")
for t in T:
    r = stat(Z[t]); out[str(t)] = r.tolist()
    print(f"  {str(t):9s}", " ".join(f"{x:5.2f}" for x in r[:8]), " || ", " ".join(f"{x:5.2f}" for x in r[8:]))
json.dump({"labels": lab, "resp": out}, open(__file__.replace(".py", ".json"), "w"))
