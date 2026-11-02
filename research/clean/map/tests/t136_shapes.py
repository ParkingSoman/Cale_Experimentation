"""b12 n136 follow-up: single straight lines failed (max 33%). Clutter of dark lines on light reached 58%.
Which local line event drives it? Draw each event once, dark (40) on light (190), width 6, centred.
PREDICTION: a sharp bend / curve in a dark stroke (hook, U-turn, loop) beats a straight stroke by >2x and reaches
>50% of max; line ends and X crossings stay below the bend."""
from lib4 import *
from PIL import Image, ImageDraw
import math
b, n = 12, 136; MX = 2.83; rng = np.random.default_rng(0)
fg, bg, w = (40, 40, 40), (190, 190, 190), 6
def mk(f):
    P = Image.new("RGB", (224, 224), bg); f(ImageDraw.Draw(P)); return np.array(P)
C = {
 "straight 45": lambda D: D.line([(40, 184), (184, 40)], fill=fg, width=w),
 "short seg 40px": lambda D: D.line([(98, 126), (126, 98)], fill=fg, width=w),
 "X crossing": lambda D: (D.line([(40, 184), (184, 40)], fill=fg, width=w), D.line([(40, 40), (184, 184)], fill=fg, width=w)),
 "T junction": lambda D: (D.line([(40, 112), (184, 112)], fill=fg, width=w), D.line([(112, 112), (112, 200)], fill=fg, width=w)),
 "L corner 90": lambda D: D.line([(60, 60), (112, 160), (190, 120)], fill=fg, width=w, joint="curve"),
 "V acute 30": lambda D: D.line([(80, 30), (112, 170), (144, 30)], fill=fg, width=w),
 "U-turn r20": lambda D: (D.arc([92, 92, 132, 132], 180, 360, fill=fg, width=w), D.line([(92, 112), (92, 200)], fill=fg, width=w), D.line([(132, 112), (132, 200)], fill=fg, width=w)),
 "circle r15": lambda D: D.ellipse([97, 97, 127, 127], outline=fg, width=w),
 "circle r30": lambda D: D.ellipse([82, 82, 142, 142], outline=fg, width=w),
 "circle r60": lambda D: D.ellipse([52, 52, 172, 172], outline=fg, width=w),
 "spiral": lambda D: D.line([(112 + 6 * t * math.cos(t), 112 + 6 * t * math.sin(t)) for t in np.linspace(0.5, 4 * math.pi, 200)], fill=fg, width=w),
 "wavy line": lambda D: D.line([(x, 112 + 25 * math.sin(x / 12)) for x in range(10, 214, 2)], fill=fg, width=w),
 "filled dot r6": lambda D: D.ellipse([106, 106, 118, 118], fill=fg),
 "dots grid": lambda D: [D.ellipse([x - 5, y - 5, x + 5, y + 5], fill=fg) for x in range(20, 224, 30) for y in range(20, 224, 30)],
 "scroll (curly)": lambda D: [D.arc([x, y, x + 40, y + 40], s, s + 270, fill=fg, width=w) for x, y, s in [(50, 50, 0), (110, 60, 90), (70, 120, 200), (130, 130, 300)]],
}
nm_, I = list(C), np.stack([mk(f) for f in C.values()])
Z = zmap(b, n, I); Zi = zmap(b, n, 255 - I)
for k, z, zi in zip(nm_, Z, Zi): print(f"{k:16s} {int(100*z.max()/MX):4d}% (inverted {int(100*zi.max()/MX):4d}%)")
grid([overlay(im, z) for im, z in zip(I, Z)], nm_, "b12_n136_shapes_heat.png", title="136 line events, dark on light")
