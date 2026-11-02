"""S8 early band synthetic battery: targets + all 45 candidates. Saves max and min over the central map region
as fractions of each neuron's natural max (pos) / |natural min| (neg) from map/deep/*.json."""
import sys, os, json, numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from early_lib import *
HERE = os.path.dirname(os.path.abspath(__file__)); DEEP = os.path.join(HERE, "..", "..", "deep")
C = json.load(open(os.path.join(HERE, "..", "..", "s8", "candidates.json")))
TG = [(4, 187), (2, 332), (3, 103), (5, 373), (1, 61)]
ALL = []
for t in TG:
    for x in [t] + [tuple(v) for v in C[f"{t[0]}_{t[1]}"].values()]:
        if x not in ALL: ALL.append(x)
NAT = {x: json.load(open(os.path.join(DEEP, f"b{x[0]}_n{x[1]}.json"))) for x in ALL}
stim, lab = [], []
def add(im, l): stim.append(np.asarray(im, np.uint8)); lab.append(l)
G = 128
for g in (0, 64, 128, 192, 255): add(patch((g,) * 3), f"flat{g}")
for name, rgb in dict(red=(200, 30, 30), green=(30, 160, 40), blue=(30, 60, 200), yellow=(230, 210, 40), cyan=(40, 200, 210),
                      magenta=(200, 40, 200), orange=(230, 120, 20), brown=(120, 70, 30), skin=(220, 170, 140)).items():
    add(patch(rgb), f"flat-{name}")
yy, xx = np.mgrid[0:224, 0:224].astype(np.float32) - 111.5
for ang in range(0, 360, 45):           # ramp: brightness rises toward direction ang (0 = right, 90 = up)
    a = np.deg2rad(ang); u = xx * np.cos(a) - yy * np.sin(a)
    for span, nm in ((224, "soft"), (64, "steep")):
        v = np.clip(128 + 255 * u / span, 0, 255); add(np.repeat(v[..., None], 3, -1), f"ramp-{nm}-{ang}")
for a in range(0, 180, 15):
    for w in (1, 3):
        add(line_img(a, w, 0, 200), f"dline-w{w}-{a}"); add(line_img(a, w, 200, 0), f"lline-w{w}-{a}")
    s = xx * np.sin(np.deg2rad(a)) + yy * np.cos(np.deg2rad(a))
    e = np.where(s > 0, 0, 200); add(np.repeat(e[..., None], 3, -1), f"edgeA-{a}"); add(np.repeat(200 - e[..., None], 3, -1), f"edgeB-{a}")
    for p in (4, 8, 16, 32): add(grating(a, p, 0.5, 0, 200), f"grat{p}-{a}")
    add(grating(a, 32, 0.5, 0, 200, square=False), f"sine32-{a}")
rng = np.random.default_rng(0)
for gr in (1, 2, 4, 8):
    n = rng.integers(0, 2, (224 // gr + 1, 224 // gr + 1)).repeat(gr, 0).repeat(gr, 1)[:224, :224] * 200
    add(np.repeat(n[..., None], 3, -1), f"noise{gr}")
    ck = (((np.arange(224)[:, None] // gr) + (np.arange(224)[None] // gr)) % 2) * 200
    add(np.repeat(ck[..., None], 3, -1), f"check{gr}")
for amp in (5, 20):
    add(np.repeat(np.clip(128 + rng.normal(0, amp, (224, 224)), 0, 255)[..., None], 3, -1), f"gnoise{amp}")
def draw(fn, bg=200):
    im = Image.new("RGB", (224, 224), (bg,) * 3); fn(ImageDraw.Draw(im)); return im
for ang in range(0, 360, 45):           # half line ending at the centre, coming from direction ang
    a = np.deg2rad(ang); x2, y2 = 112 + 150 * np.cos(a), 112 - 150 * np.sin(a)
    add(draw(lambda d: d.line([(112, 112), (x2, y2)], fill=(0,) * 3, width=3)), f"lineend-{ang}")
for ang in (0, 90, 180, 270):            # L corner at centre: arms toward ang and ang+90
    a, b = np.deg2rad(ang), np.deg2rad(ang + 90)
    add(draw(lambda d: d.line([(112 + 150 * np.cos(a), 112 - 150 * np.sin(a)), (112, 112), (112 + 150 * np.cos(b), 112 - 150 * np.sin(b))], fill=(0,) * 3, width=3, joint=None)), f"corner-{ang}")
add(draw(lambda d: (d.line([(0, 112), (224, 112)], fill=(0,) * 3, width=3), d.line([(112, 0), (112, 224)], fill=(0,) * 3, width=3))), "cross")
add(draw(lambda d: (d.line([(0, 112), (224, 112)], fill=(0,) * 3, width=3), d.line([(112, 112), (112, 224)], fill=(0,) * 3, width=3))), "Tjunc")
add(draw(lambda d: d.ellipse([106, 106, 118, 118], fill=(0,) * 3)), "dot12")
add(draw(lambda d: d.ellipse([109, 109, 115, 115], fill=(0,) * 3)), "dot6")
add(draw(lambda d: d.ellipse([92, 92, 132, 132], outline=(0,) * 3, width=3)), "ring40")
add(draw(lambda d: d.rectangle([108, 60, 116, 164], fill=(0,) * 3)), "bar8-vert")
add(draw(lambda d: d.rectangle([60, 108, 164, 116], fill=(0,) * 3)), "bar8-horz")
try: font = ImageFont.load_default(size=22)
except TypeError: font = ImageFont.load_default()
txt = "I can't help\nyour tran-\nsaction to\nday. 1980\nBAROMETER\nHALO OF D"
add(draw(lambda d: d.multiline_text((10, 20), txt, fill=(0,) * 3, font=font, spacing=8), 235), "text-dark")
add(draw(lambda d: d.multiline_text((10, 20), txt, fill=(235,) * 3, font=font, spacing=8), 20), "text-light")
X = np.stack(stim); Z = multi(ALL, X)
R = {}
for t in ALL:
    k = int(Z[t].shape[1] * 0.25); zz = Z[t][:, k:-k, k:-k]
    R[f"{t[0]},{t[1]}"] = {"max": (zz.max((1, 2)) / NAT[t]["max"]).round(3).tolist(), "min": (zz.min((1, 2)) / abs(NAT[t]["min"])).round(3).tolist()}
json.dump({"labels": lab, "resp": R}, open(os.path.join(HERE, "battery.json"), "w"))
print(len(lab), "stimuli,", len(ALL), "neurons")
