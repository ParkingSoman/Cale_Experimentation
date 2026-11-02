"""T4 (8 n4), follow-up after T3 falsified 'long horizontal edge'. T3: full-width step -1.2, but a step bounded
left and right by gray (56 px wide) 2.33. PREDICTION (before run): 8 n4 needs a horizontal edge that ENDS at vertical
edges (corners / T-junctions of a box). A filled 56x56 square on gray scores > 2.0; a grid of thin lines (collage
seams, 56 px cells) scores > 2.0; the same grid rotated 45 deg scores < 1.0; a lone vertical bar scores < 0."""
from mid_common import *
from PIL import ImageFilter
S = 224; T = [(8, 4), (11, 676), (11, 745), (11, 318), (5, 652), (5, 642), (6, 205), (6, 459)]


def C(v=124): return Image.new("L", (S, S), v)


def rect(w, h, fill=40, outline=None, bg=124):
    im = C(bg); d = ImageDraw.Draw(im); l, t = (S - w) // 2, (S - h) // 2
    d.rectangle([l, t, l + w, t + h], fill=fill, outline=outline, width=3 if outline is not None else 0); return im


def grid(cell, lw=3, val=30, rot=0):
    im = C(); d = ImageDraw.Draw(im)
    for k in range(-S, 2 * S, cell): d.rectangle([k, -S, k + lw - 1, 2 * S], fill=val); d.rectangle([-S, k, 2 * S, k + lw - 1], fill=val)
    return im.rotate(rot, fillcolor=124, resample=Image.BILINEAR)


def shapes(kind, val=30):
    im = C(); d = ImageDraw.Draw(im); c = S // 2
    if kind == "plus": d.rectangle([c - 60, c - 1, c + 60, c + 1], fill=val); d.rectangle([c - 1, c - 60, c + 1, c + 60], fill=val)
    if kind == "T": d.rectangle([c - 60, c - 1, c + 60, c + 1], fill=val); d.rectangle([c - 1, c, c + 1, c + 60], fill=val)
    if kind == "T_up": d.rectangle([c - 60, c - 1, c + 60, c + 1], fill=val); d.rectangle([c - 1, c - 60, c + 1, c], fill=val)
    if kind == "L": d.rectangle([c, c - 1, c + 60, c + 1], fill=val); d.rectangle([c - 1, c, c + 1, c + 60], fill=val)
    if kind == "vbar": d.rectangle([c - 28, 0, c + 28, S], fill=val)
    if kind == "hbar": d.rectangle([0, c - 28, S, c + 28], fill=val)
    if kind == "comb":
        d.rectangle([20, c - 40, S - 20, c], fill=220)
        for x in range(20, S - 20, 12): d.rectangle([x, c, x + 8, c + 50], fill=230)
    if kind == "disk": d.ellipse([c - 28, c - 28, c + 28, c + 28], fill=val)
    return im


P = {}
for w in (16, 32, 56, 96, 160): P[f"dark sq {w}"] = rect(w, w)
for w in (32, 56, 96): P[f"light sq {w}"] = rect(w, w, fill=220)
P["dark rect 160x40"] = rect(160, 40); P["dark rect 40x160"] = rect(40, 160)
P["outline sq 96"] = rect(96, 96, fill=124, outline=30); P["dark sq 56 rot45"] = rect(56, 56).rotate(45, fillcolor=124)
P["dark sq 56 blur2"] = rect(56, 56).filter(ImageFilter.GaussianBlur(2)); P["disk 56"] = shapes("disk")
for c in (28, 56, 112): P[f"grid {c}"] = grid(c)
P["grid 56 rot45"] = grid(56, rot=45); P["grid 56 white"] = grid(56, val=240)
for k in ("plus", "T", "T_up", "L", "vbar", "hbar", "comb"): P[k] = shapes(k)
P["gray"] = C()
names = list(P); imgs = np.stack([np.array(P[k].convert("RGB")) for k in names]); Z = zmaps(imgs, T)
print("probe".ljust(20) + "".join(f"{b}n{n}".rjust(9) for b, n in T))
for i, k in enumerate(names): print(k.ljust(20) + "".join(f"{Z[t][i].max():9.2f}" for t in T))
print(sheet([(heat(imgs[i], Z[(8, 4)][i]), f"{k} {Z[(8, 4)][i].max():.2f}") for i, k in enumerate(names)], "t4_n4_probes.png", ncol=9, size=100, title="8 n4 junction probes"))
