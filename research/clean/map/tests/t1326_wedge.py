"""b14 n1326 shape test. From the roll / reflect-shift heat maps: red sits on straight oblique edges that close off
a wedge (two straight edges meeting at a point), often with the image frame as one side.
PREDICTION (before running): a flat-coloured wedge with an acute apex (<=60 deg), drawn in the interior, drives the
unit to >50% of its max (>3.0) near the apex; a plain straight edge (half-plane), a disc of similar area, and a
blank canvas stay below 25% of max (<1.5). Polarity (light-on-dark vs dark-on-light) does not matter (invert 0.99)."""
from lib4 import *
from PIL import Image, ImageDraw
import math
MX = 6.47; b, n = 14, 1326
def canvas(fg, bg, draw):
    P = Image.new("RGB", (224, 224), bg); draw(ImageDraw.Draw(P), fg); return np.array(P)
def wedge(ang, cx=112, cy=150, L=110, rot=0):
    def f(D, fg):
        a0 = math.radians(-90 + rot - ang / 2); a1 = math.radians(-90 + rot + ang / 2)
        D.polygon([(cx, cy), (cx + L * math.cos(a0), cy + L * math.sin(a0)), (cx + L * math.cos(a1), cy + L * math.sin(a1))], fill=fg)
    return f
cases = {}
for ang in (20, 40, 60, 90, 120, 150):
    cases[f"wedge {ang}deg"] = wedge(ang)
cases["wedge 40 rot45"] = wedge(40, rot=45, cx=80, cy=150)
cases["half-plane diag"] = lambda D, fg: D.polygon([(0, 224), (224, 0), (224, 224)], fill=fg)
cases["half-plane vert"] = lambda D, fg: D.rectangle([112, 0, 224, 224], fill=fg)
cases["disc r40"] = lambda D, fg: D.ellipse([72, 72, 152, 152], fill=fg)
cases["square 80"] = lambda D, fg: D.rectangle([72, 72, 152, 152], fill=fg)
cases["square 80 rot45 (diamond)"] = lambda D, fg: D.polygon([(112, 55), (169, 112), (112, 169), (55, 112)], fill=fg)
cases["frame corner cut (triangle at corner)"] = lambda D, fg: D.polygon([(0, 0), (70, 0), (0, 70)], fill=fg)
cases["thin line diag"] = lambda D, fg: D.line([(30, 194), (194, 30)], fill=fg, width=3)
cases["blank"] = lambda D, fg: None
names_, imgs = [], []
for k, f in cases.items():
    for pol, (fg, bg) in (("L/D", ((220, 220, 220), (60, 60, 60))), ("D/L", ((60, 60, 60), (220, 220, 220)))):
        names_.append(f"{k} {pol}"); imgs.append(canvas(fg, bg, f))
Z = zmap(b, n, np.stack(imgs))
for nmx, z in zip(names_, Z):
    p = np.unravel_index(np.argmax(z), z.shape); print(f"{nmx:42s} max {z.max():5.2f} ({100*z.max()/MX:4.0f}% of max) at {p}")
grid([overlay(im, z) for im, z in zip(imgs, Z)], names_, "b14_n1326_wedge_heat.png", cols=8, title="1326 synthetic shapes")
