"""Do b3 n623's parent candidate (2 n61) and readers share its tuning? Flat gray, isoluminant blue-yellow noise,
red-green noise and luminance noise (grain 8 px, amplitude 40), and saturated multi-hue noise. Raw z (central max)."""
import json, colorsys, numpy as np
from PIL import Image
from early_lib import *
from t_chroma import rgb
T = [(3, 623), (2, 61), (4, 467), (4, 395), (4, 401), (0, 346), (0, 364)]
rs = np.random.default_rng(4)
def nz(g=8): return np.array(Image.fromarray(rs.choice([0, 255], (224 // g, 224 // g)).astype(np.uint8)).resize((224, 224), Image.NEAREST)) / 127.5 - 1
stim, lab = [], []
c = np.zeros((224, 224, 3), np.float32); c[..., 0] = 128; stim.append(rgb(c)); lab.append("flat gray")
for ax, k in (("lum", 0), ("by", 1), ("rg", 2)):
    c = np.zeros((224, 224, 3), np.float32); c[..., 0] = 128; c[..., k] += 40 * nz(); stim.append(rgb(c)); lab.append(f"{ax} noise")
blk = rs.integers(0, 12, (28, 28)); hues = np.array([colorsys.hsv_to_rgb(h / 12, .8, .8) for h in range(12)]) * 255
stim.append(np.array(Image.fromarray(hues[blk].astype(np.uint8)).resize((224, 224), Image.NEAREST))); lab.append("multi-hue blocks")
Z = multi(T, np.stack(stim)); out = {}
print("cols:", lab)
for t in T:
    r = stat(Z[t]); out[str(t)] = r.tolist(); print(f"  {str(t):9s}", " ".join(f"{x:6.2f}" for x in r))
json.dump({"labels": lab, "resp": out}, open(__file__.replace(".py", ".json"), "w"))
