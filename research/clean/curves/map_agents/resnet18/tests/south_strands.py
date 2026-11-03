# serves: H-map
"""South: drawn test images. Thin long strands vs short dense strokes vs solid blobs."""
import math, os, random
from PIL import Image, ImageDraw
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "south_imgs")
os.makedirs(D, exist_ok=True)
S = 224
random.seed(0)
def new(bg=(200, 190, 170)):
    im = Image.new("RGB", (S, S), bg); return im, ImageDraw.Draw(im)
im, d = new((60, 70, 90)); d.ellipse((20, 20, 204, 204), fill=(240, 240, 240))
d.line((112, 112, 112, 45), fill=(0, 0, 0), width=3); d.line((112, 112, 165, 130), fill=(0, 0, 0), width=3)
im.save(f"{D}/clock.png")
im, d = new()
for i in range(120):
    x = random.randint(0, S); pts = [(x + 12 * math.sin(t / 25 + i), t) for t in range(0, S, 4)]
    d.line(pts, fill=(250, 245, 235), width=1)
im.save(f"{D}/long_strands.png")
im, d = new((120, 80, 50))
for i in range(2500):
    x, y = random.randint(0, S), random.randint(0, S); d.line((x, y, x + 3, y + 3), fill=(90, 55, 30), width=1)
im.save(f"{D}/short_coat.png")
im, d = new((200, 190, 170)); d.ellipse((50, 50, 180, 180), fill=(110, 70, 40)); im.save(f"{D}/solid_blob.png")
im, d = new((240, 240, 240))
for a in range(0, 360, 20): d.line((112, 112, 112 + 100 * math.cos(math.radians(a)), 112 + 100 * math.sin(math.radians(a))), fill=(0, 0, 0), width=2)
im.save(f"{D}/spokes.png")
im, d = new((60, 70, 90)); d.ellipse((20, 20, 204, 204), fill=(240, 240, 240)); im.save(f"{D}/disc_nohands.png")
