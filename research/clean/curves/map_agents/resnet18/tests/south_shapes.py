# serves: H-map
"""South: long tapered wedge vs short blunt blob (muzzle length dial); vertical vs horizontal drape."""
import math, os
from PIL import Image, ImageDraw
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "south_imgs")
os.makedirs(D, exist_ok=True)
S = 224
def new(bg=(120, 150, 100)):
    im = Image.new("RGB", (S, S), bg); return im, ImageDraw.Draw(im)
im, d = new(); d.polygon([(20, 112), (170, 95), (170, 130)], fill=(150, 90, 50)); d.ellipse((150, 70, 215, 155), fill=(150, 90, 50)); im.save(f"{D}/wedge_long.png")
im, d = new(); d.polygon([(120, 105), (170, 95), (170, 130), (120, 125)], fill=(150, 90, 50)); d.ellipse((130, 60, 215, 165), fill=(150, 90, 50)); im.save(f"{D}/wedge_short.png")
def drape(vertical):
    im, d = new((200, 200, 205))
    for k in range(14):
        x = 20 + k * 14; c = (150 + (k % 3) * 20, 40, 60 + (k % 2) * 40)
        pts = [(x + 6 * math.sin(t / 15), t) for t in range(10, 215, 5)]
        d.line(pts, fill=c, width=12)
    return im if vertical else im.rotate(90)
drape(True).save(f"{D}/drape_vert.png"); drape(False).save(f"{D}/drape_horiz.png")
