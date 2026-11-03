# serves: H-map
import os, math
from PIL import Image, ImageDraw
out = "/private/tmp/claude-501/east/"
os.makedirs(out, exist_ok=True)

def sky(c=(150, 190, 235)):
    im = Image.new("RGB", (224, 224), c); return im, ImageDraw.Draw(im)

# 277: smooth grey dome on grass vs spoked wheel
im, d = sky(); d.rectangle([0, 160, 224, 224], fill=(60, 140, 60)); d.pieslice([30, 60, 194, 260], 180, 360, fill=(215, 215, 220)); im.save(out + "dome_grass.png")
im, d = sky((235, 235, 235)); d.ellipse([40, 40, 184, 184], outline=(30, 30, 30), width=6)
for a in range(0, 360, 30):
    d.line([(112, 112), (112 + 72 * math.cos(math.radians(a)), 112 + 72 * math.sin(math.radians(a)))], fill=(30, 30, 30), width=3)
im.save(out + "spoked.png")
# 403: bundle of parallel dark tubes vs concentric circles
im, d = sky((200, 200, 200))
for i in range(6):
    d.rectangle([20, 40 + i * 28, 204, 40 + i * 28 + 16], fill=(35, 35, 40))
im.save(out + "tubes.png")
im, d = sky((235, 235, 235))
for r in (90, 70, 50, 30):
    d.ellipse([112 - r, 112 - r, 112 + r, 112 + r], outline=(30, 30, 30), width=5)
im.save(out + "rings.png")
# 367: dark rounded hood around a lighter face oval vs wedge
im, d = sky((220, 220, 220)); d.ellipse([50, 30, 174, 200], fill=(30, 30, 40)); d.ellipse([80, 70, 144, 150], fill=(225, 190, 160)); im.save(out + "hood.png")
im, d = sky((220, 220, 220)); d.polygon([(112, 30), (190, 200), (112, 170), (34, 200)], fill=(30, 30, 40)); im.save(out + "wedge.png")
