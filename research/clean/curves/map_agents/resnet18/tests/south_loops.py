# serves: H-map
"""South: drawn chain loops, curls, brush bristles, straight lines, orange blobs."""
import math, os, random
from PIL import Image, ImageDraw
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "south_imgs")
os.makedirs(D, exist_ok=True)
S = 224
random.seed(1)
def new(bg=(30, 30, 40)):
    im = Image.new("RGB", (S, S), bg); return im, ImageDraw.Draw(im)
im, d = new()                                     # chain: interlocking rings along a circle
for k in range(14):
    a = 2 * math.pi * k / 14; cx, cy = 112 + 75 * math.cos(a), 112 + 75 * math.sin(a)
    d.ellipse((cx - 17, cy - 17, cx + 17, cy + 17), outline=(210, 210, 215), width=4)
im.save(f"{D}/chain.png")
im, d = new((200, 190, 170))                       # curls: many spirals
for k in range(25):
    cx, cy = random.randint(20, 204), random.randint(20, 204)
    pts = [(cx + (t / 3) * math.cos(t / 2), cy + (t / 3) * math.sin(t / 2)) for t in range(0, 40)]
    d.line(pts, fill=(120, 80, 40), width=3)
im.save(f"{D}/curls.png")
im, d = new((200, 190, 170))                       # straight parallel lines
for x in range(10, 220, 12): d.line((x, 10, x, 214), fill=(120, 80, 40), width=3)
im.save(f"{D}/straight_lines.png")
im, d = new((235, 235, 235))                       # brush: handle + fanned bristles
d.rectangle((100, 140, 124, 220), fill=(90, 50, 30))
for i in range(60):
    x0 = 100 + i * 0.4; d.line((x0, 140, 60 + i * 1.7, 40 + random.randint(0, 10)), fill=(200, 90, 100), width=2)
im.save(f"{D}/brush.png")
for k, c in {"orange": (230, 110, 30), "blue": (40, 70, 200), "black2": (15, 15, 15), "white2": (250, 250, 250)}.items():
    im, d = new((130, 130, 130)); d.ellipse((60, 70, 170, 170), fill=c); d.ellipse((130, 45, 180, 90), fill=c)
    im.save(f"{D}/col_{k}.png")
