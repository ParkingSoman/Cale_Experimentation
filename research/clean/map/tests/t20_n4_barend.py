"""T20 (8 n4). T4/T5: tall 40x160 rect 3.30 but wide 160x40 rect 1.00; full-width edges fail. New reading: the
square-cut END of an upright bar or panel. PREDICTION (before run): a 40 px wide dark bar from the top edge down to
the centre (one square end) > 2.5; the same bar with a ROUNDED end < 1.0; a single filled corner (dark quadrant, one
horizontal edge meeting one vertical edge) > 1.5. Width sweep of a 160 px tall bar: no prediction."""
from mid_common import *
S = 224; t = (8, 4)


def im(draw):
    p = Image.new("L", (S, S), 124); draw(ImageDraw.Draw(p)); return p


P = {"bar from top, square end": im(lambda d: d.rectangle([92, -5, 132, 112], fill=40)),
     "bar from top, round end": im(lambda d: (d.rectangle([92, -5, 132, 92], fill=40), d.ellipse([92, 72, 132, 112], fill=40))),
     "bar from bottom, square end": im(lambda d: d.rectangle([92, 112, 132, 230], fill=40)),
     "bar from left, square end": im(lambda d: d.rectangle([-5, 92, 112, 132], fill=40)),
     "corner quadrant (dark TL)": im(lambda d: d.rectangle([-5, -5, 112, 112], fill=40)),
     "corner quadrant (dark BR)": im(lambda d: d.rectangle([112, 112, 230, 230], fill=40)),
     "light bar from top, square": im(lambda d: d.rectangle([92, -5, 132, 112], fill=220))}
for w in (8, 16, 40, 80, 120, 160): P[f"bar w{w} h160"] = im(lambda d, w=w: d.rectangle([112 - w // 2, 32, 112 + w // 2, 192], fill=40))
names = list(P); imgs = np.stack([np.array(P[k].convert("RGB")) for k in names]); Z = zmaps(imgs, [t, (6, 205), (5, 652)])
print("probe".ljust(30) + "8n4".rjust(7) + "6n205".rjust(8) + "5n652".rjust(8))
for i, k in enumerate(names): print(k.ljust(30) + f"{Z[t][i].max():7.2f}{Z[(6, 205)][i].max():8.2f}{Z[(5, 652)][i].max():8.2f}")
print(sheet([(heat(imgs[i], Z[t][i]), f"{k[:20]} {Z[t][i].max():.2f}") for i, k in enumerate(names)], "t20_n4.png", ncol=7, size=110, title="8 n4 bar ends"))
