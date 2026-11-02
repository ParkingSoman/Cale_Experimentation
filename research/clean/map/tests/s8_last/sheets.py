"""Tile rows of existing deep-view PNGs (target + its 9 candidates) into sheets for blind judging."""
import json, os, sys
from PIL import Image, ImageDraw
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
C = json.load(open(os.path.join(D, "s8", "candidates.json")))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)
ROWS = {"100": 26, "75": 26 + 104, "50": 26 + 208, "neg": 26 + 5 * 104}
import sys
for key, cands in [(k, C[k]) for k in sys.argv[1:]]:
    b, n = key.split("_"); items = [("T", int(b), int(n))] + [(c, *v) for c, v in cands.items()]
    for part, sl in (("a", items[:5]), ("b", [items[0]] + items[5:])):
        H = len(sl) * (3 * 92 + 16) + 4; sheet = Image.new("RGB", (660, H), "white"); d = ImageDraw.Draw(sheet); y = 2
        for tag, bb, nn in sl:
            im = Image.open(os.path.join(D, "deep", f"b{bb}_n{nn}.png")); ed = json.load(open(os.path.join(D, "deep", f"b{bb}_n{nn}.json")))["edits"]
            d.text((4, y), f"{tag}: b{bb} n{nn}  gray {ed['gray']:.2f} blur3 {ed['blur3']:.2f} rot90 {ed['rot90']:.2f} shuf8 {ed['shuf8']:.2f} hue {ed['hue']:.2f} zoomout {ed['zoomout']:.2f}", fill="blue"); y += 14
            for r in ("100", "75", "neg"):
                sheet.paste(im.crop((0, ROWS[r] + 12, 660, ROWS[r] + 12 + 80)), (0, y)); y += 92 - 8
            y += 10
        sheet.save(os.path.join(OUT, f"s8_{key}_{part}.png"))
print("ok")
