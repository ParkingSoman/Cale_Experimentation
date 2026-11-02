"""Show the 96x96 window at the peak of a neuron on its top analysis images of given classes. usage: B N fname classes"""
import sys, numpy as np
from PIL import Image, ImageDraw
from last_lib import *
b, n, fname, cls = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4].split(",")
X, y, ana = data(); nm = names(); ya = y[ana]; v = amax(b, n); idx = []
for c in cls:
    j = np.where(ya == nm.index(c))[0]; idx += [int(ana[k]) for k in j[np.argsort(-v[j])[:6]]]
im = np.stack([np.asarray(X[i]) for i in idx]); Z = zmaps(b, n, im); sh = Image.new("RGB", (100 * 6, 112 * ((len(idx) + 5) // 6)), "white"); d = ImageDraw.Draw(sh)
for j in range(len(idx)):
    r, c = np.unravel_index(Z[j].argmax(), (7, 7)); t, l = int(np.clip((r - 1) * 32, 0, 128)), int(np.clip((c - 1) * 32, 0, 128))
    sh.paste(Image.fromarray(im[j, t:t + 96, l:l + 96]), (100 * (j % 6), 112 * (j // 6))); d.text((100 * (j % 6), 112 * (j // 6) + 97), f"{nm[y[idx[j]]][:10]} {Z[j].max():.1f}", fill="black")
sh.save(f"figs/{fname}"); print(f"figs/{fname}")
