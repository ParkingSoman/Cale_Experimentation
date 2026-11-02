import json, os
import numpy as np
from PIL import Image, ImageDraw
from gallery_x import *
import gallery_x as gx
A1 = json.load(open(os.path.join(RES, "vision-a1-small.json"))); DC = json.load(open(os.path.join(RES, "vision-a1c-decode.json")))
ids = [tuple(x) for x in A1["ids"]["stage4"]]; r2 = np.array(DC["atto"]["stage4_r2"]); arg = np.array(A1["atto"]["best_match"]["stage4"]["arg"]); rb = np.array(A1["atto"]["best_match"]["stage4"]["r"])
MAP = json.load(open(os.path.join(ROOT, "research/clean/map/network_map_v2.json")))["neurons"]; kids = {}
for k, v in MAP.items(): b, n = map(int, k.split("_")); kids.setdefault(tuple(v["parent"]), []).append((b, n))
desc = {}
for b in range(17, -1, -1):
    for n in range(3072 if b >= 15 else 1536 if b >= 6 else 768): desc[(b, n)] = sum(1 + desc[c] for c in kids.get((b, n), []))
d = np.array([desc[k] for k in ids]); rr = np.random.default_rng(4)
leaf = np.where((d == 0) & (r2 < .02))[0]; trunk = np.where((d > 10) & (r2 > .3))[0]
pick = [("LEAF", j) for j in rr.choice(leaf, 3, replace=False)] + [("TRUNK", j) for j in rr.choice(trunk, 3, replace=False)]
want = {}
for _, j in pick: want.setdefault(ids[j][0], []).append(ids[j][1])
Ht = maps("tiny", want); W = [40 * 4 * 2 * 2] ; aw = {}
for _, j in pick:
    a = int(arg[j]); b = 10
    while a >= 1280: a -= 1280; b += 1
    aw.setdefault(b, []).append(a)
Ha = maps("atto", aw); names = [n.split(",")[0] for n in class_names()]; k = 8
img = Image.new("RGB", (T * k + 16, (T + 30) * 2 * len(pick) + 40), "white"); dr = ImageDraw.Draw(img); yy = 34
dr.text((8, 6), "Last stage. LEAF = no map descendants, atto cannot rebuild it (R^2 < .02). TRUNK = >10 descendants, atto rebuilds it (R^2 > .3)", fill="black")
for kind, j in pick:
    a = int(arg[j]); b = 10
    while a >= 1280: a -= 1280; b += 1
    for cap, h in ((f"{kind}: tiny block {ids[j][0]} n{ids[j][1]} ({d[j]} descendants, atto decode R^2 {r2[j]:.2f})", Ht[ids[j]]), (f"   its best atto single match: block {b} n{a} (r {rb[j]:.2f})", Ha[(b, a)])):
        dr.text((8, yy - 12), cap, fill="black")
        for q, (i, p) in enumerate(spots(h, k)): img.paste(crop(i, p, 3), (8 + q * T, yy)); dr.text((8 + q * T, yy + T), names[gx._y[gx._idx[i]]][:12], fill="darkgreen")
        yy += T + 30
fp = os.path.join(HERE, "figs", "a1", "leaves_vs_trunks.png"); img.save(fp); print(fp)
