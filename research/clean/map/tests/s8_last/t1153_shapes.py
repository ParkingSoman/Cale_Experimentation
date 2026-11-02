"""15 n1153 prediction: the lower convex rim of a rounded mass hanging above open space (bowl underside).
A bowl shape (curved side down) in the upper half on sky gives >30% of max; a dome (curved side up) gives
less than half of the bowl; the same bowl near the bottom edge gives less."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_lib import sky, run, table
from PIL import ImageDraw
T = (15, 1153); C2 = (14, 1101); C6 = (10, 1066); C5 = (14, 707)
ref = {T: 9.87, C2: 3.6, C6: 3.4, C5: 3.4}
S = {}
for col, cn in (((90, 95, 100), "dark"), ((235, 235, 230), "light")):
    for w in (70, 120):
        h = w // 2; cx, cy = 112, 70
        im = sky(); d = ImageDraw.Draw(im); d.pieslice([cx - w // 2, cy - h, cx + w // 2, cy + h], 0, 180, fill=col); S[f"bowl{w}_{cn}"] = im
        im = sky(); d = ImageDraw.Draw(im); d.pieslice([cx - w // 2, cy - h + 30, cx + w // 2, cy + h + 30], 180, 360, fill=col); S[f"dome{w}_{cn}"] = im
        im = sky(); d = ImageDraw.Draw(im); d.ellipse([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2], fill=col); S[f"ellipse{w}_{cn}"] = im
        im = sky(); d = ImageDraw.Draw(im); d.ellipse([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2], fill=col)
        for x in (cx - w // 3, cx - w // 8, cx + w // 8, cx + w // 3): d.line([x, cy + h // 3, x + (x - cx) // 3, 224], fill=(60, 60, 60), width=2)
        S[f"tower{w}_{cn}"] = im
        im = sky(); d = ImageDraw.Draw(im); d.pieslice([cx - w // 2, 200 - h, cx + w // 2, 200 + h], 0, 180, fill=col); S[f"bowlLow{w}_{cn}"] = im
        im = sky(); d = ImageDraw.Draw(im); d.rectangle([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2], fill=col); S[f"rect{w}_{cn}"] = im
S["blank"] = sky()
out, Z, names = run(S, [T, C2, C6, C5]); table(out, [T, C2, C6, C5], ref)
