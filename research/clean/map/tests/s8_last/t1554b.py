"""15 n1554 follow-up (after t1554: upward spray failed, rot180 hit 68%): leaves that HANG down from a point.
Prediction: a downward fan of pointed leaves gives >50% of max in gray too; a downward fan of round dots or thin
lines, or a blob, gives <20%; an upward fan stays low."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_lib import sky, run, table
from t1554 import spray, leaf, tg, ref
from PIL import Image, ImageDraw, ImageOps
S = {}
down = lambda **k: spray(cy=94, **k).rotate(180)
S["hang dark"] = down(); S["hang gray"] = ImageOps.grayscale(down()).convert("RGB"); S["hang green"] = down(col=(60, 140, 40))
S["hang 12"] = down(n=12); S["hang 3"] = down(n=3); S["hang big"] = down(L=95, W=14); S["hang small"] = down(L=30, W=5)
S["hang narrow fan"] = down(spread=math.pi / 3)
S["side fan (right)"] = spray(cx=100, cy=112).rotate(-90)
im = sky(); d = ImageDraw.Draw(im)
for i in range(7):
    a = math.pi * i / 6; d.ellipse([112 + 40 * math.cos(a) - 9, 94 + 40 * math.sin(a) - 9, 112 + 40 * math.cos(a) + 9, 94 + 40 * math.sin(a) + 9], fill=(40, 50, 30))
S["hang dots"] = im
im = sky(); d = ImageDraw.Draw(im)
for i in range(7): d.line([112, 94, 112 + 60 * math.cos(math.pi * i / 6), 94 + 60 * math.sin(math.pi * i / 6)], fill=(40, 50, 30), width=3)
S["hang thin lines"] = im
im = sky(); d = ImageDraw.Draw(im)
for x in range(30, 200, 28): leaf(d, x, 20, math.pi / 2 + .25 * ((x // 28) % 2 - .5), 70, 9, (40, 50, 30))
S["row of hanging leaves"] = im
out, Z, n = run(S, tg); table(out, tg, ref)
