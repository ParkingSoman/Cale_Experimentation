"""T3 (8 n4). PREDICTION (written before the run): a single sharp, full-width horizontal step edge on gray drives
8 n4 above 2.0 (50% of max 4.02); the same edge rotated 90 deg stays below 0; an edge only 1/4 of the width long
drops below half of the full-width value. Extra sweep (no prediction): square-wave stripes, period x angle; single
thin lines; blur. Also runs the same probes on readers 11 n676, 11 n745, 11 n318 and inputs 5 n652, 5 n642, 6 n205."""
from mid_common import *
from PIL import ImageFilter
S = 224; T = [(8, 4), (11, 676), (11, 745), (11, 318), (5, 652), (5, 642), (6, 205), (6, 459)]


def canvas(v=124): return Image.new("L", (S, S), v)


def step(angle=0, frac=1.0, lo=60, hi=190):
    a = np.full((S, S), lo, np.uint8); a[:S // 2] = hi
    if frac < 1: w = int(S * frac); l = (S - w) // 2; a[:, :l] = 124; a[:, l + w:] = 124
    return Image.fromarray(a).rotate(angle, fillcolor=124, resample=Image.BILINEAR)


def line(angle=0, frac=1.0, width=2, val=30):
    im = canvas(); d = ImageDraw.Draw(im); w = int(S * frac); l = (S - w) // 2
    d.rectangle([l, S // 2 - width // 2, l + w, S // 2 + (width - 1) // 2], fill=val); return im.rotate(angle, fillcolor=124, resample=Image.BILINEAR)


def stripes(period, angle):
    yy, xx = np.mgrid[:S, :S] - S / 2; t = np.deg2rad(angle); u = yy * np.cos(t) + xx * np.sin(t)
    return Image.fromarray(np.where((u % period) < period / 2, 60, 190).astype(np.uint8))


probes = {}
for ang in (0, 15, 30, 45, 60, 90): probes[f"step a{ang}"] = step(ang)
for fr in (1.0, .5, .25, .125): probes[f"step len{fr}"] = step(0, fr)
probes["step blur1"] = step().filter(ImageFilter.GaussianBlur(1)); probes["step blur3"] = step().filter(ImageFilter.GaussianBlur(3))
probes["step inverted"] = step(lo=190, hi=60)
for ang in (0, 45, 90): probes[f"line a{ang}"] = line(ang)
for fr in (1.0, .5, .25): probes[f"line len{fr}"] = line(0, fr)
probes["line white"] = line(val=230); probes["line w6"] = line(width=6)
for p in (4, 6, 8, 12, 16, 24, 32, 48):
    for ang in (0, 30, 90): probes[f"stripe p{p} a{ang}"] = stripes(p, ang)
probes["stripe p8 a0 blur1"] = stripes(8, 0).filter(ImageFilter.GaussianBlur(1))
probes["gray"] = canvas()
names = list(probes); imgs = np.stack([np.array(probes[k].convert("RGB")) for k in names])
Z = zmaps(imgs, T); mxs = {(8, 4): 4.02}
print("probe".ljust(22) + "".join(f"{b}n{n}".rjust(9) for b, n in T))
for i, k in enumerate(names): print(k.ljust(22) + "".join(f"{Z[t][i].max():9.2f}" for t in T))
cells = [(heat(imgs[i], Z[(8, 4)][i]), f"{k} {Z[(8, 4)][i].max():.2f}") for i, k in enumerate(names)]
print(sheet(cells[:24], "t3_n4_probes.png", ncol=8, size=110, title="8 n4 on probes"))
