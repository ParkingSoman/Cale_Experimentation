"""T13 (10 n1316). From T12: top cells have a BRIGHTER region below them and are darker themselves.
PREDICTION (before run): 10 n1316 marks the lower boundary of a darker mass sitting above a brighter field
(dark-above / bright-below). A dark 80x120 rectangle on a 220 background: the cells at its BOTTOM edge reach > 50% of
max (2.4) and at least 2x the cells at its TOP edge. A bright rectangle on a dark background fires at its TOP edge
instead. Full-frame two-tone (dark top half, bright bottom) > reverse two-tone."""
from mid_common import *
from PIL import ImageFilter
S = 224; t = (10, 1316); T = [t, (1, 80), (0, 106), (5, 221), (4, 721), (9, 916), (7, 604)]


def scene(bg, obj, w=80, h=120, shape="rect", cx=112, cy=112):
    im = Image.new("L", (S, S), bg); d = ImageDraw.Draw(im); box = [cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2]
    if obj is not None: (d.rectangle if shape == "rect" else d.ellipse)(box, fill=obj)
    return im


def two(top, bot):
    a = np.full((S, S), bot, np.uint8); a[:S // 2] = top; return Image.fromarray(a)


P = {}
for bg in (240, 200, 124, 60):
    P[f"bg{bg} none"] = scene(bg, None); P[f"bg{bg} dark rect"] = scene(bg, 40); P[f"bg{bg} bright rect"] = scene(bg, 235)
P["bg220 dark ellipse"] = scene(220, 40, 100, 120, "ellipse"); P["bg220 dark rect blur3"] = scene(220, 40).filter(ImageFilter.GaussianBlur(3))
P["bg220 dark rect low"] = scene(220, 40, cy=140); P["bg220 dark rect high"] = scene(220, 40, cy=80)
P["two dark/bright"] = two(40, 220); P["two bright/dark"] = two(220, 40); P["two mid/bright"] = two(124, 220)
names = list(P); imgs = np.stack([np.array(P[k].convert("RGB")) for k in names]); Z = zmaps(imgs, T)
print("probe".ljust(24) + "max".rjust(7) + "  bottom-edge  top-edge  left-edge | other neurons (max):" + "".join(f"{b}n{n}".rjust(8) for b, n in T[1:]))
for i, k in enumerate(names):
    z = Z[t][i]; a = np.array(P[k]); rows = np.where((a < 100).any(1) if "dark" in k else (a > 230).any(1))[0]
    if "rect" in k or "ellipse" in k:
        r_top, r_bot = rows.min() // 16, rows.max() // 16; cols = slice(5, 9)
        be, te_, le = z[min(r_bot, 13), cols].max(), z[max(r_top, 0), cols].max(), z[r_top:r_bot + 1, 3:5].max()
    else: be = te_ = le = np.nan
    print(k.ljust(24) + f"{z.max():7.2f}{be:12.2f}{te_:10.2f}{le:10.2f} |" + "".join(f"{Z[q][i].max():8.2f}" for q in T[1:]))
print(sheet([(heat(imgs[i], Z[t][i]), f"{k} {Z[t][i].max():.2f}") for i, k in enumerate(names)], "t13_n1316.png", ncol=9, size=100, title="10 n1316 probes"))
