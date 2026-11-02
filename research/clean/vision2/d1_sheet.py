"""Crop sheet: first row = a small-model neuron's strongest spots; each next row = a tiny neuron's strongest spots, with
the small neuron's value there (% of its max) printed under each crop. Also the small neuron's most negative spots.
usage: python d1_sheet.py SMALL BLOCK NEURON tinyB:n tinyB:n ... (fname from args)"""
import sys
import numpy as np
from PIL import Image, ImageDraw
import gallery_x as G
from convnext import *
small, sb, sn = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); tiny = [tuple(int(v) for v in a.split(":")) for a in sys.argv[4:]]
G._init(); names = [n.split(",")[0] for n in class_names()]; k = 8; T = G.T
want = {}
for b, n in tiny: want.setdefault(b, []).append(n)
Ht = G.maps("tiny", want); Hs = G.maps(small, {sb: [sn]})[(sb, sn)]; st = G.model(small).stage_of[sb]; smx = Hs.max()
rows = [(f"{small} {sb}/{sn}: strongest", G.spots(Hs, k), Hs, None), (f"{small} {sb}/{sn}: most negative", G.spots(-Hs, k), Hs, None)]
for b, n in tiny:
    h = Ht[(b, n)]; c = np.corrcoef(h.reshape(-1), Hs.reshape(-1))[0, 1]
    rows.append((f"tiny {b}/{n} (corr {c:.2f}): strongest; green = tiny %, blue = {small} %", G.spots(h, k), h, Hs))
W = T * k + 16; Hh = (T + 30) * len(rows) + 20; img = Image.new("RGB", (W, Hh), "white"); dr = ImageDraw.Draw(img); yy = 22
for cap, sp, val, other in rows:
    dr.text((8, yy - 13), cap, fill="black"); mv = max(np.abs(val).max(), 1e-6)
    for j, (i, q) in enumerate(sp):
        img.paste(G.crop(i, q, st), (8 + j * T, yy)); dr.text((8 + j * T, yy + T), f"{names[G._y[G._idx[i]]][:8]} {100 * val[i, q] / mv:.0f}", fill="darkgreen")
        if other is not None: dr.text((8 + j * T + 62, yy + T), f"{100 * other[i, q] / smx:.0f}", fill="blue")
    yy += T + 30
fn = f"d/sheet_{small}_{sb}_{sn}.png"; fp = os.path.join(HERE, "figs", fn); img.save(fp); print(fp)
