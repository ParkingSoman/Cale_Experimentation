"""b1 n351: lines / edges / gratings at angles. Target + parents (0 n245, 0 n193, cand B 0 n22) + readers."""
import json, numpy as np
from PIL import Image, ImageDraw, ImageFilter
from early_lib import *

T = [(1, 351), (0, 245), (0, 193), (0, 22), (3, 754), (4, 731)]
NAT = {(1, 351): 5.49, (0, 245): 8.35, (0, 193): 10.42, (0, 22): 5.04}
angles = list(range(0, 180, 15)); stim, lab = [], []


def edge(angle, dark_side):
    yy, xx = np.mgrid[0:224, 0:224].astype(np.float32) - 111.5; a = np.deg2rad(angle)
    s = xx * np.sin(a) + yy * np.cos(a); v = np.where((s > 0) == dark_side, 0, 200)
    return np.repeat(v[..., None], 3, -1).astype(np.uint8)


for a in angles:
    for w in (1, 2, 4, 8):
        stim.append(line_img(a, w, 0, 200)); lab.append(("dark-line", a, w))
        stim.append(line_img(a, w, 200, 0)); lab.append(("light-line", a, w))
    stim.append(edge(a, True)); lab.append(("edge+", a, 0)); stim.append(edge(a, False)); lab.append(("edge-", a, 0))
    for p in (6, 12, 24): stim.append(grating(a, p, duty=0.25, lo=200, hi=0)); lab.append((f"grat{p}", a, 0))
# controls
blur = np.array(Image.fromarray(line_img(90, 2, 0, 200)).filter(ImageFilter.GaussianBlur(3))); stim.append(blur); lab.append(("blur3-vline", 90, 2))
mesh = np.minimum(grating(45, 12, .2, 200, 0), grating(135, 12, .2, 200, 0)); stim.append(mesh); lab.append(("mesh45", 45, 0))
cross = np.minimum(grating(0, 12, .2, 200, 0), grating(90, 12, .2, 200, 0)); stim.append(cross); lab.append(("mesh0-90", 0, 0))
for g in (0, 100, 200): stim.append(patch((g, g, g))); lab.append((f"flat{g}", 0, 0))
# color line on colored ground (gray edit said color-blind)
stim.append(np.array(Image.fromarray(patch((40, 120, 220))).copy())); lab.append(("blue-flat", 0, 0))
im = Image.fromarray(patch((230, 200, 60))); ImageDraw.Draw(im).line([(111, 0), (111, 224)], fill=(30, 60, 200), width=2); stim.append(np.array(im)); lab.append(("blue-line-on-yellow", 90, 2))
Z = multi(T, np.stack(stim)); R = {f"{t[0]},{t[1]}": stat(Z[t]).tolist() for t in T}
json.dump({"labels": lab, "resp": R}, open(__file__.replace(".py", ".json"), "w"))
r = np.array(R["1,351"]) / NAT[(1, 351)]
print("b1 n351 as fraction of natural max (5.49). rows = angle (0=horizontal, 90=vertical)")
kinds = ["dark-line", "light-line"]
for k in kinds:
    print(k, "widths 1/2/4/8")
    for a in angles:
        print(f"  {a:3d}", " ".join(f"{r[i]:6.2f}" for i, l in enumerate(lab) if l[0] == k and l[1] == a))
print("edges and gratings: angle edge+ edge- grat6 grat12 grat24")
for a in angles:
    print(f"  {a:3d}", " ".join(f"{r[i]:6.2f}" for i, l in enumerate(lab) if l[1] == a and l[0] in ("edge+", "edge-", "grat6", "grat12", "grat24")))
for i, l in enumerate(lab):
    if l[0] in ("blur3-vline", "mesh45", "mesh0-90", "flat0", "flat100", "flat200", "blue-flat", "blue-line-on-yellow"): print(l[0], f"{r[i]:.2f}")
# parents: vertical vs horizontal tuning
for t in T[1:]:
    rr = np.array(R[f"{t[0]},{t[1]}"]); nat = NAT.get(t, rr.max())
    sel = lambda k, a, w: rr[[i for i, l in enumerate(lab) if l == (k, a, w)][0]] / nat
    print(t, "dark v-line w2 %.2f  light v-line w2 %.2f  dark h-line w2 %.2f  dark 45 w2 %.2f  edge+90 %.2f edge-90 %.2f grat12-90 %.2f" % (
        sel("dark-line", 90, 2), sel("light-line", 90, 2), sel("dark-line", 0, 2), sel("dark-line", 45, 2), sel("edge+", 90, 0), sel("edge-", 90, 0), sel("grat12", 90, 0)),
        "(raw units)" if t not in NAT else "")
