"""15 n1153 follow-up: which parts of the cartoon water tower matter (tank, legs, pole), for target and candidates."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_lib import sky, run, table
from PIL import ImageDraw
T = (15, 1153); C2 = (14, 1101); C6 = (10, 1066); C5 = (14, 707); C8 = (13, 1460); C4 = (14, 1411)
tg = [T, C2, C6, C5, C8, C4]; ref = {T: 9.87, C2: 3.6, C6: 3.4, C5: 3.4, C8: 3.7, C4: 3.8}
col = (90, 95, 100); cx, cy, w = 112, 70, 70; h = w // 2
def tank(d): d.ellipse([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2], fill=col)
def legs(d):
    for x in (cx - w // 3, cx - w // 8, cx + w // 8, cx + w // 3): d.line([x, cy + h // 3, x + (x - cx) // 3, 224], fill=(60, 60, 60), width=2)
S = {}
im = sky(); d = ImageDraw.Draw(im); tank(d); legs(d); S["tank+legs"] = im
im = sky(); d = ImageDraw.Draw(im); legs(d); S["legs only"] = im
im = sky(); d = ImageDraw.Draw(im); tank(d); d.rectangle([cx - 4, cy, cx + 4, 224], fill=(60, 60, 60)); S["tank+pole"] = im
im = sky(); d = ImageDraw.Draw(im); tank(d); d.rectangle([0, cy + h // 2 + 2, 224, 224], fill=(110, 140, 80)); S["tank over ground"] = im
im = sky(); d = ImageDraw.Draw(im); d.ellipse([cx - 60, cy - 8, cx + 60, cy + 8], fill=col); S["thin disk edge-on"] = im
im = sky(); d = ImageDraw.Draw(im); tank(d); im = im.rotate(180); S["tank+legs? no: tank rot180"] = im
im = sky(); d = ImageDraw.Draw(im); tank(d); legs(d); S["tower rot180"] = im.rotate(180)
im = sky(); d = ImageDraw.Draw(im); tank(d); legs(d); S["tower rot90"] = im.rotate(90)
im = sky((235, 235, 235), (235, 235, 235)); d = ImageDraw.Draw(im); tank(d); legs(d); S["tower on white"] = im
out, Z, names = run(S, tg); table(out, tg, ref)
