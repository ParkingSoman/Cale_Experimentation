# serves: H-map
import numpy as np, os
from PIL import Image, ImageDraw
out = "/private/tmp/claude-501/east/"
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(1)

def canvas(bg=(225, 225, 225)):
    im = Image.new("RGB", (224, 224), bg)
    return im, ImageDraw.Draw(im)

# green floret cluster vs green smooth blob
im, d = canvas((240, 240, 240))
for _ in range(60):
    x, y = rng.integers(50, 175, 2); r = rng.integers(9, 18)
    d.ellipse([x - r, y - r, x + r, y + r], fill=(30, int(rng.integers(90, 150)), 30))
im.save(out + "florets.png")
im, d = canvas((240, 240, 240)); d.ellipse([50, 50, 175, 175], fill=(40, 120, 40)); im.save(out + "green_blob.png")
# teardrop (body tapering to point) vs circle, brown
im, d = canvas(); d.ellipse([40, 70, 150, 160], fill=(130, 80, 40)); d.polygon([(130, 75), (210, 120), (130, 155)], fill=(130, 80, 40)); im.save(out + "teardrop.png")
im, d = canvas(); d.ellipse([60, 60, 160, 160], fill=(130, 80, 40)); im.save(out + "brown_circle.png")
# dome vs thin rod, dark
im, d = canvas(); d.pieslice([30, 90, 190, 250], 180, 360, fill=(60, 50, 40)); im.save(out + "dome.png")
im, d = canvas(); d.rectangle([20, 105, 205, 118], fill=(60, 50, 40)); im.save(out + "rod.png")
