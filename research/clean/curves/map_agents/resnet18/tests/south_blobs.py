# serves: H-map
"""South: same blob shape, different colours, on a mid-grey or green ground. Is the node a pale-body dial?"""
import os
from PIL import Image, ImageDraw
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "south_imgs")
os.makedirs(D, exist_ok=True)
S = 224
cols = {"white": (250, 250, 250), "cream": (235, 220, 180), "yellow": (240, 210, 40), "brown": (110, 70, 40),
        "black": (20, 20, 20), "red": (200, 30, 30)}
for ground, g in (("grey", (130, 130, 130)), ("green", (70, 120, 60))):
    for k, c in cols.items():
        im = Image.new("RGB", (S, S), g); d = ImageDraw.Draw(im)
        d.ellipse((60, 70, 170, 170), fill=c); d.ellipse((130, 45, 180, 90), fill=c)   # body + head
        im.save(f"{D}/blob_{k}_{ground}.png")
