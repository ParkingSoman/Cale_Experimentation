"""Montage of deep views (title + 100% row + 50% row + strongest-negative row + heat row) for several neurons."""
import sys, os
from PIL import Image, ImageDraw
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deep"); OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
names = sys.argv[2:]; parts = []
for nm in names:
    im = Image.open(os.path.join(D, nm + ".png")); W = im.width
    rows = [im.crop((0, 0, W, 26))] + [im.crop((0, 26 + 104 * k, W, 26 + 104 * k + 104)) for k in (0, 2, 5)] + [im.crop((0, 26 + 104 * 7, W, 26 + 104 * 7 + 126))]
    h = sum(r.height for r in rows); c = Image.new("RGB", (W, h + 14), "white"); dd = ImageDraw.Draw(c); dd.text((4, 0), nm, fill="blue"); y = 12
    for r in rows: c.paste(r, (0, y)); y += r.height
    parts.append(c)
H = sum(p.height for p in parts); M = Image.new("RGB", (max(p.width for p in parts), H), "white"); y = 0
for p in parts: M.paste(p, (0, y)); y += p.height
M.save(os.path.join(OUT, sys.argv[1])); print(os.path.join(OUT, sys.argv[1]))
