"""17 n966 follow-up: which part of the clasp frame matters (arc, knobs), on or off a pouch; orientation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_lib import run, table
from PIL import Image, ImageDraw
T = (17, 966); tg = [T, (16, 1205), (15, 506), (16, 442)]; ref = {T: 5.55, (16, 1205): 1.09, (15, 506): 0.84, (16, 442): 1.0}
def draw(pouch=True, arc=True, knobs=True, bg=(225, 222, 215)):
    im = Image.new("RGB", (224, 224), bg); d = ImageDraw.Draw(im)
    if pouch: d.rectangle([40, 80, 184, 150], fill=(200, 60, 90)); d.ellipse([40, 100, 184, 200], fill=(200, 60, 90))
    if arc: d.arc([40, 50, 184, 110], 180, 360, fill=(170, 170, 175), width=6)
    if knobs: d.ellipse([100, 38, 112, 50], fill=(190, 190, 195)); d.ellipse([112, 38, 124, 50], fill=(190, 190, 195))
    return im
S = {"full": draw(), "pouch+arc": draw(knobs=False), "pouch+knobs": draw(arc=False), "arc+knobs no pouch": draw(pouch=False),
     "arc only": draw(pouch=False, knobs=False), "full rot180": draw().rotate(180), "full rot90": draw().rotate(90),
     "full gray bg dark": draw(bg=(60, 60, 60)), "full small": draw().resize((112, 112)).crop((0, 0, 112, 112))}
im = Image.new("RGB", (224, 224), (225, 222, 215)); im.paste(draw().resize((112, 112)), (56, 56)); S["full small"] = im
im = Image.new("RGB", (224, 224), (225, 222, 215)); d = ImageDraw.Draw(im); d.rectangle([40, 80, 184, 150], fill=(200, 60, 90)); d.ellipse([40, 100, 184, 200], fill=(200, 60, 90)); d.rectangle([40, 74, 184, 82], fill=(170, 170, 175)); S["pouch+straight bar"] = im
out, Z, n = run(S, tg); table(out, tg, ref)
