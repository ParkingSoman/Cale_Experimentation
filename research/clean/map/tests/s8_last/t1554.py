"""15 n1554 prediction: a spray of pointed leaves (several narrow blades radiating from one point, gaps between them),
shape not colour. A dark leaf spray on sky gives >30% of max (6.48); a solid blob of equal area and a cluster of round
dots give less than half of the spray; gray copy keeps the value. Same stimuli on C3, C4, C6, C8, C2."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_lib import sky, run, table
from PIL import Image, ImageDraw, ImageOps
T = (15, 1554); tg = [T, (13, 1520), (12, 1448), (12, 59), (14, 822), (14, 301)]
ref = {T: 6.48, (13, 1520): 6.0, (12, 1448): 4.29, (12, 59): 5.58, (14, 822): 7.09, (14, 301): 5.0}
def leaf(d, cx, cy, ang, L, W, col):
    pts = []
    for t in range(0, 21):
        u = t / 20; r = W * math.sin(math.pi * u)
        pts.append((u * L, r))
    pts = pts + [(x, -r) for x, r in reversed(pts)]
    ca, sa = math.cos(ang), math.sin(ang)
    d.polygon([(cx + x * ca - y * sa, cy + x * sa + y * ca) for x, y in pts], fill=col)
def spray(n=7, L=60, W=9, col=(40, 50, 30), cx=112, cy=130, spread=math.pi, bg=None):
    im = bg or sky(); d = ImageDraw.Draw(im)
    for i in range(n): leaf(d, cx, cy, -math.pi / 2 - spread / 2 + spread * i / (n - 1), L, W, col)
    return im
S = {}
S["spray dark"] = spray(); S["spray green"] = spray(col=(60, 140, 40)); S["spray red"] = spray(col=(170, 50, 30))
S["spray dark gray"] = ImageOps.grayscale(spray()).convert("RGB")
S["spray small"] = spray(L=30, W=5); S["spray big"] = spray(L=95, W=14)
S["spray 12 leaves"] = spray(n=12); S["spray 3 leaves"] = spray(n=3)
S["spray rot180"] = spray().rotate(180); S["spray rot90"] = spray().rotate(90)
im = sky(); d = ImageDraw.Draw(im); d.ellipse([75, 75, 149, 140], fill=(40, 50, 30)); S["blob equal area"] = im
im = sky(); d = ImageDraw.Draw(im)
for i in range(7):
    a = -math.pi + math.pi * i / 6; d.ellipse([112 + 40 * math.cos(a) - 9, 130 + 40 * math.sin(a) - 9, 112 + 40 * math.cos(a) + 9, 130 + 40 * math.sin(a) + 9], fill=(40, 50, 30))
S["round dots ring"] = im
im = sky(); d = ImageDraw.Draw(im)
for i in range(7): d.line([112, 130, 112 + 60 * math.cos(-math.pi + math.pi * i / 6), 130 + 60 * math.sin(-math.pi + math.pi * i / 6)], fill=(40, 50, 30), width=3)
S["thin lines fan"] = im
S["spray on white wall"] = spray(bg=Image.new("RGB", (224, 224), (230, 228, 220)))
S["blank sky"] = sky()
out, Z, n = run(S, tg); table(out, tg, ref)
