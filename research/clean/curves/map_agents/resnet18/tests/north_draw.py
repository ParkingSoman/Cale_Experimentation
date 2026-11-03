# serves: H-map
import math
import sys

from PIL import Image, ImageDraw, ImageFilter

out = sys.argv[1]
S = 224


def base(c):
    return Image.new("RGB", (S, S), c)


for handle in (1, 0):
    im = base((235, 235, 235))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((60, 70, 140, 170), 10, fill=(250, 250, 250), outline=(170, 170, 170), width=3)
    if handle:
        d.arc((125, 85, 185, 150), -90, 90, fill=(170, 170, 170), width=10)
    im.save(f"{out}/mug_{handle}.png")

for name, col in (("plume_white", (245, 245, 250)), ("plume_dark", (60, 60, 60))):
    im = Image.new("RGB", (S, S))
    d = ImageDraw.Draw(im)
    for y in range(S):
        d.line((0, y, S, y), fill=(90, 140, 210) if y < 150 else (120, 110, 90))
    p = Image.new("RGB", (S, S), (0, 0, 0))
    pd = ImageDraw.Draw(p)
    pd.polygon([(95, 150), (125, 150), (140, 40), (80, 40)], fill=col)
    pd.ellipse((60, 10, 160, 80), fill=col)
    m = p.convert("L").point(lambda v: 255 if v > 0 else 0).filter(ImageFilter.GaussianBlur(6))
    im.paste(p, (0, 0), m)
    im.save(f"{out}/{name}.png")

im = base((150, 190, 235)); d = ImageDraw.Draw(im)
d.rectangle((0, 150, S, S), fill=(90, 110, 70)); d.polygon([(10, 150), (112, 50), (214, 150)], fill=(110, 110, 115))
im.filter(ImageFilter.GaussianBlur(2)).save(f"{out}/mountain.png")
im = base((150, 190, 235)); d = ImageDraw.Draw(im)
d.rectangle((0, 150, S, S), fill=(90, 110, 70)); d.ellipse((95, 110, 130, 145), fill=(110, 110, 115))
im.save(f"{out}/blob.png")

im = base((240, 240, 240)); d = ImageDraw.Draw(im)
d.ellipse((70, 50, 150, 130), outline=(60, 60, 60), width=6); d.line((110, 130, 110, 200), fill=(60, 60, 60), width=6)
im.save(f"{out}/loop.png")
im = base((240, 240, 240)); d = ImageDraw.Draw(im)
for k in range(16):
    a = k * math.pi / 8
    d.line((112, 112, 112 + 80 * math.cos(a), 112 + 80 * math.sin(a)), fill=(60, 60, 60), width=4)
im.save(f"{out}/spokes.png")

im = base((235, 235, 235)); d = ImageDraw.Draw(im)
d.ellipse((30, 85, 120, 140), fill=(120, 30, 40)); d.rectangle((120, 108, 200, 116), fill=(200, 160, 90))
im.save(f"{out}/ball_rod.png")
im = base((235, 235, 235)); d = ImageDraw.Draw(im)
d.rectangle((30, 100, 200, 125), fill=(120, 30, 40))
im.save(f"{out}/bar.png")
