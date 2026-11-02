"""15 n2818 prediction: a local part at the lower body of a boxy machine, where a flat panel meets a wheel or opening.
Keeping a 96 px window at the peak (rest gray) keeps >50% of the original max (median over top-20). Moving the
window 48 px up (more panel, less wheel) or 48 px down drops it below half of the centred window. A cartoon
box with two wheels gives >25% of max (4.34); the box alone and the wheels alone give less. Same stimuli on C1, C7, C9, C3."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from win_lib import top_images, peak_px, window, maps
from synth_lib import sky, run, table
from PIL import Image, ImageDraw
T = (15, 2818); tg = [T, (12, 1190), (12, 1514), (14, 550), (14, 92)]; tags = ["T", "C1", "C7", "C9", "C3"]
imgs, idx = top_images(*T, k=20); Z = maps(imgs, [T])[T]; P = [peak_px(z) for z in Z]
E = {"orig": imgs, "win peak": np.stack([window(im, cy, cx) for im, (cy, cx) in zip(imgs, P)]),
     "win up48": np.stack([window(im, cy - 48, cx) for im, (cy, cx) in zip(imgs, P)]),
     "win down48": np.stack([window(im, cy + 48, cx) for im, (cy, cx) in zip(imgs, P)]),
     "win left48": np.stack([window(im, cy, cx - 48) for im, (cy, cx) in zip(imgs, P)]),
     "gray peak window": np.stack([np.where(window(np.zeros_like(im), cy, cx, w=64, gray=1) == 0, 128, im).astype(np.uint8) for im, (cy, cx) in zip(imgs, P)])}
A = {k: maps(v, tg) for k, v in E.items()}; ref = {t: np.median(A["orig"][t].max((1, 2))) for t in tg}
print("median image max (ratio to orig)")
for k, d in A.items(): print(k.ljust(18) + "  ".join(f"{g} {np.median(d[t].max((1, 2))):6.2f} ({np.median(d[t].max((1, 2)))/ref[t]:+.2f})" for g, t in zip(tags, tg)))
print("peak rows (px) of top-20:", [p[0] for p in P])
import sys as _s
if len(_s.argv) > 1: raise SystemExit
# cartoons
S = {}
def ground(): im = sky(); d = ImageDraw.Draw(im); d.rectangle([0, 170, 224, 224], fill=(110, 110, 105)); return im, d
for c, cn in (((170, 40, 30), "red"), ((90, 100, 60), "olive")):
    im, d = ground(); d.rectangle([40, 90, 190, 160], fill=c); d.rectangle([140, 60, 185, 95], fill=c); d.ellipse([55, 135, 95, 175], fill=(25, 25, 25)); d.ellipse([140, 135, 180, 175], fill=(25, 25, 25)); S[f"truck {cn}"] = im
    im, d = ground(); d.rectangle([40, 90, 190, 160], fill=c); d.rectangle([140, 60, 185, 95], fill=c); S[f"box only {cn}"] = im
im, d = ground(); d.ellipse([55, 135, 95, 175], fill=(25, 25, 25)); d.ellipse([140, 135, 180, 175], fill=(25, 25, 25)); S["wheels only"] = im
im, d = ground(); d.rectangle([30, 40, 194, 200], fill=(200, 200, 205)); d.ellipse([72, 80, 152, 160], fill=(40, 40, 50)); S["washer"] = im
out, Zs, nms = run(S, tg); table(out, tg, {t: ref[t] for t in tg})
