"""17 n966 prediction: the job sits on the top rim of a soft patterned pouch, where a thin metal frame (purse clasp)
borders a finely patterned surface. (a) Real images (top-20): keeping a 96 px window at the peak keeps >50%;
graying only a 64 px patch at the peak drops it below 50%; blurring only that patch (sigma 3) drops it below 50%.
(b) Cartoon: a pouch with stripes + metal arc frame gives more than the same pouch without frame, and more than a
flat-colour pouch with frame."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from win_lib import top_images, peak_px, window, maps
from synth_lib import run, table
from PIL import Image, ImageDraw, ImageFilter
T = (17, 966); tg = [T, (16, 1205), (15, 506), (16, 442)]; tags = ["T", "C1", "C7", "C9"]
imgs, idx = top_images(*T, k=20); Z = maps(imgs, [T])[T]; P = [peak_px(z) for z in Z]
def gpatch(im, cy, cx, w=64, mode="gray"):
    out = im.copy(); y0 = int(np.clip(cy - w // 2, 0, 224 - w)); x0 = int(np.clip(cx - w // 2, 0, 224 - w))
    if mode == "gray": out[y0:y0 + w, x0:x0 + w] = 128
    else: out[y0:y0 + w, x0:x0 + w] = np.asarray(Image.fromarray(im).filter(ImageFilter.GaussianBlur(3)))[y0:y0 + w, x0:x0 + w]
    return out
far = lambda cy, cx: ((cy + 112) % 224, (cx + 112) % 224)
E_ = {"orig": imgs, "win peak": np.stack([window(im, *p) for im, p in zip(imgs, P)]),
      "gray patch peak": np.stack([gpatch(im, *p) for im, p in zip(imgs, P)]),
      "gray patch far": np.stack([gpatch(im, *far(*p)) for im, p in zip(imgs, P)]),
      "blur patch peak": np.stack([gpatch(im, *p, mode="blur") for im, p in zip(imgs, P)]),
      "blur whole": np.stack([np.asarray(Image.fromarray(im).filter(ImageFilter.GaussianBlur(3))) for im in imgs])}
A = {k: maps(v, tg) for k, v in E_.items()}; ref = {t: np.median(A["orig"][t].max((1, 2))) for t in tg}
print("median image max (ratio to orig)")
for k, d in A.items(): print(k.ljust(18) + "  ".join(f"{g} {np.median(d[t].max((1, 2))):6.2f} ({np.median(d[t].max((1, 2)))/ref[t]:+.2f})" for g, t in zip(tags, tg)))
print("peak (row,col) px:", P)
S = {}
def pouch(pattern, frame, bg=(225, 222, 215)):
    im = Image.new("RGB", (224, 224), bg); d = ImageDraw.Draw(im)
    m = Image.new("L", (224, 224), 0); dm = ImageDraw.Draw(m); dm.rectangle([40, 80, 184, 150], fill=255); dm.ellipse([40, 100, 184, 200], fill=255)
    tex = Image.new("RGB", (224, 224), (200, 60, 90)); dt = ImageDraw.Draw(tex)
    if pattern == "stripes":
        for yy in range(0, 224, 8): dt.rectangle([0, yy, 224, yy + 3], fill=(30, 30, 80))
    if pattern == "dots":
        for yy in range(0, 224, 12):
            for xx in range(0, 224, 12): dt.ellipse([xx, yy, xx + 5, yy + 5], fill=(250, 240, 200))
    im.paste(tex, (0, 0), m); d = ImageDraw.Draw(im)
    if frame:
        d.arc([40, 50, 184, 110], 180, 360, fill=(170, 170, 175), width=6); d.ellipse([100, 38, 112, 50], fill=(190, 190, 195)); d.ellipse([112, 38, 124, 50], fill=(190, 190, 195))
    return im
for p in ("stripes", "dots", "flat"):
    for f in (1, 0): S[f"pouch {p} frame{f}"] = pouch(p, f)
tex = pouch("dots", 0).crop((60, 120, 160, 180)).resize((224, 224)); S["dots texture only"] = tex
out, Zs, nms = run(S, tg); table(out, tg, ref)
