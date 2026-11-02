"""Pooled (mean, 14x14) maps of units on the b9 dark-rectangle stimulus and its inverse. Usage: python rectmaps.py B,N ..."""
import sys, os
import numpy as np
from PIL import Image, ImageDraw
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "s6"))
from judge_lib import maps
U = [tuple(map(int, a.split(","))) for a in sys.argv[1:]]
im = Image.new("RGB", (224, 224), (190,) * 3); ImageDraw.Draw(im).rectangle([40, 70, 184, 150], fill=(30,) * 3)
X = np.stack([np.asarray(im), 255 - np.asarray(im)]); Z = maps(X, U); np.set_printoptions(linewidth=200, precision=1, suppress=True)
ref = None
for u in U:
    z = Z[u]; f = z.shape[1] // 14; z = z if f == 1 else z.reshape(2, 14, f, 14, f).mean((2, 4))
    if ref is None: ref = z
    r = [np.corrcoef(ref[k].ravel(), z[k].ravel())[0, 1] for k in range(2)]
    print(f"\n== b{u[0]}n{u[1]}  corr with first unit: rect {r[0]:+.2f}  inverse {r[1]:+.2f}"); print(z[0][::2, ::2])
