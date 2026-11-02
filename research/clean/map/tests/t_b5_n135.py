"""b5 n135: orientation x polarity x scale. Gratings (periods 4..32), step edges (both polarities), thin lines,
wide bars. Target + inputs (4 n212, 4 n27, 3 n315, cand B 2 n173, neg 4 n320, 3 n660) + readers (8 n106, 7 n60, 6 n808).
Also b4 n499: colour boundary (blue on one side, yellow on the other) rotated through 360 deg."""
import json, numpy as np
from early_lib import *
from t_chroma import rgb

T = [(5, 135), (4, 212), (4, 27), (3, 315), (2, 173), (4, 320), (3, 660), (8, 106), (7, 60), (6, 808)]
NAT = {(5, 135): 3.67}
angles = list(range(0, 180, 15)); stim, lab = [], []


def edge(angle, flip, lo=0, hi=200):
    yy, xx = np.mgrid[0:224, 0:224].astype(np.float32) - 111.5; a = np.deg2rad(angle)
    s = xx * np.sin(a) + yy * np.cos(a); v = np.where((s > 0) != flip, hi, lo)
    return np.repeat(v[..., None], 3, -1).astype(np.uint8)


for a in angles:
    for p in (4, 8, 16, 32): stim.append(grating(a, p, .5, 0, 200)); lab.append((f"grat{p}", a))
    stim.append(edge(a, False)); lab.append(("edgeA", a)); stim.append(edge(a, True)); lab.append(("edgeB", a))
    stim.append(line_img(a, 2, 0, 200)); lab.append(("dline", a)); stim.append(line_img(a, 2, 200, 0)); lab.append(("lline", a))
    stim.append(line_img(a, 8, 0, 200)); lab.append(("dbar8", a)); stim.append(line_img(a, 8, 200, 0)); lab.append(("lbar8", a))
Z = multi(T, np.stack(stim)); R = {f"{b},{n}": stat(Z[(b, n)]).tolist() for b, n in T}
kinds = ["grat4", "grat8", "grat16", "grat32", "edgeA", "edgeB", "dline", "lline", "dbar8", "lbar8"]
r = np.array(R["5,135"]) / 3.67
print("5,135 fraction of max 3.67; angle: 0 = horizontal stripes/line, 45 = '/' , 90 = vertical, 135 = '\\'")
print("       " + " ".join(f"{k:>6s}" for k in kinds))
for a in angles: print(f"  {a:3d}  " + " ".join(f"{r[i]:6.2f}" for i, l in enumerate(lab) if l[1] == a))
for t in T[1:]:
    rr = np.array(R[f"{t[0]},{t[1]}"]); best = int(np.argmax(rr)); worst = int(np.argmin(rr))
    print(f"{t} corr with 5,135 r={np.corrcoef(rr, r)[0,1]:+.2f}  best {lab[best]} {rr[best]:.2f}  worst {lab[worst]} {rr[worst]:.2f}")
# n499 colour boundary rotation: blue on the side 'angle' points to
yy, xx = np.mgrid[0:224, 0:224].astype(np.float32) - 111.5; cs, cl = [], []
for ang in range(0, 360, 30):
    t = np.deg2rad(ang); side = (xx * np.cos(t) - yy * np.sin(t)) > 0
    c = np.zeros((224, 224, 3), np.float32); c[..., 0] = 128; c[..., 1] = np.where(side, 30, -30); cs.append(rgb(c)); cl.append(ang)
z = stat(multi([(4, 499)], np.stack(cs))[(4, 499)]) / 4.51
print("4,499 blue/yellow boundary, blue side direction (90 = blue above, 270 = blue below):", dict(zip(cl, z.round(2))))
json.dump({"labels": lab, "resp": R, "n499_rot": dict(zip(cl, z.tolist()))}, open(__file__.replace(".py", ".json"), "w"))
