# serves: H-map
"""west: draw test images for stage-4 inner channels. Written BEFORE the probe run.
Predictions (can fail):
 P1 s4b1i:393 fires on a diamond lattice (>=30% of its largest) and far less on parallel stripes and on dots.
 P2 s4b1i:189 fires on small dark dots on a light ground (>=30%) and less on the same dots inverted or on stripes.
 P3 s4b1i:64 fires on a plain smooth flat orange patch more than on the same patch with noise added.
 P4 s4b1i:419 fires on a white blob on dark green more than on a brown blob on the same ground.
 P5 s4b1i:196 fires on a small saturated blue blob with thin legs on green more than a grey blob of the same shape.
"""
import os
import numpy as np
from PIL import Image, ImageDraw

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "west_imgs")
os.makedirs(OUT, exist_ok=True)
S = 224
rng = np.random.default_rng(0)


def save(im, n):
    im.save(os.path.join(OUT, n + ".png"))


# lattice
im = Image.new("RGB", (S, S), (150, 190, 220)); d = ImageDraw.Draw(im)
for k in range(-S, 2 * S, 28):
    d.line([(k, 0), (k + S, S)], fill=(60, 60, 60), width=3)
    d.line([(k, S), (k + S, 0)], fill=(60, 60, 60), width=3)
save(im, "lattice")
im = Image.new("RGB", (S, S), (150, 190, 220)); d = ImageDraw.Draw(im)
for k in range(0, S, 28):
    d.line([(k, 0), (k, S)], fill=(60, 60, 60), width=3)
save(im, "stripes")
# dots
im = Image.new("RGB", (S, S), (225, 225, 220)); d = ImageDraw.Draw(im)
for _ in range(60):
    x, y = rng.integers(10, S - 10, 2); r = int(rng.integers(4, 8))
    d.ellipse([x - r, y - r, x + r, y + r], fill=(30, 30, 30))
save(im, "dots_dark")
im = Image.new("RGB", (S, S), (30, 30, 30)); d = ImageDraw.Draw(im)
for _ in range(60):
    x, y = rng.integers(10, S - 10, 2); r = int(rng.integers(4, 8))
    d.ellipse([x - r, y - r, x + r, y + r], fill=(225, 225, 220))
save(im, "dots_light")
# smooth patch
im = Image.new("RGB", (S, S), (200, 200, 200)); d = ImageDraw.Draw(im)
d.rounded_rectangle([40, 60, 184, 170], 30, fill=(235, 120, 30))
save(im, "smooth_orange")
a = np.array(im).astype(float); a[60:170, 40:184] += rng.normal(0, 40, (110, 144, 3))
save(Image.fromarray(np.clip(a, 0, 255).astype("uint8")), "noisy_orange")
# white vs brown blob on dark green
for nm, col in (("white", (245, 245, 245)), ("brown", (140, 80, 40))):
    im = Image.new("RGB", (S, S), (30, 70, 35)); d = ImageDraw.Draw(im)
    d.ellipse([60, 90, 170, 170], fill=col); d.ellipse([140, 70, 190, 115], fill=col)
    save(im, nm + "_blob")
# saturated blue vs grey bird with thin legs
for nm, col in (("bluebird", (30, 70, 230)), ("greybird", (110, 110, 110))):
    im = Image.new("RGB", (S, S), (90, 140, 70)); d = ImageDraw.Draw(im)
    d.ellipse([70, 70, 150, 130], fill=col)
    d.line([(100, 125), (95, 190)], fill=(60, 50, 40), width=3); d.line([(120, 125), (125, 190)], fill=(60, 50, 40), width=3)
    save(im, nm)
# radial
im = Image.new("RGB", (S, S), (240, 240, 240)); d = ImageDraw.Draw(im)
import math
for a_ in range(0, 360, 15):
    d.line([(112, 112), (112 + 105 * math.cos(math.radians(a_)), 112 + 105 * math.sin(math.radians(a_)))], fill=(40, 40, 40), width=3)
save(im, "radial")
