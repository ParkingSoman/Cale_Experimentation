# serves: H-map
import os
from PIL import Image, ImageDraw
out = "/private/tmp/claude-501/east/"
os.makedirs(out, exist_ok=True)

def sky():
    im = Image.new("RGB", (224, 224), (150, 190, 235)); return im, ImageDraw.Draw(im)

# 344: vertical thin stick with round head vs the same rotated to horizontal
im, d = sky(); d.rectangle([108, 70, 116, 210], fill=(60, 40, 30)); d.ellipse([98, 40, 126, 72], fill=(200, 60, 40)); im.save(out + "stick_vert.png")
im, d = sky(); d.rectangle([20, 108, 160, 116], fill=(60, 40, 30)); d.ellipse([150, 98, 182, 126], fill=(200, 60, 40)); im.save(out + "stick_horiz.png")
# 329: fan of diagonal thin lines (rigging) vs fan of horizontal lines
im, d = sky()
for i in range(8):
    d.line([(40 + 20 * i, 215), (112, 10)], fill=(40, 40, 40), width=2)
im.save(out + "rigging_diag.png")
im, d = sky()
for i in range(8):
    d.line([(10, 40 + 22 * i), (214, 40 + 22 * i)], fill=(40, 40, 40), width=2)
im.save(out + "lines_horiz.png")
# 486: swept-wing plane silhouette on sky vs blob
im, d = sky(); d.polygon([(112, 60), (125, 120), (200, 150), (125, 140), (118, 190), (112, 175), (106, 190), (99, 140), (24, 150), (99, 120)], fill=(70, 70, 80)); im.save(out + "plane.png")
im, d = sky(); d.ellipse([70, 70, 150, 150], fill=(70, 70, 80)); im.save(out + "blob.png")
