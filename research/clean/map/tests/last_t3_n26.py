"""16 n26 test: peaked canopy (apex-up striped triangle / cone) vs apex-down vs rectangle, synthetic."""
import numpy as np
from PIL import Image, ImageDraw
from last_lib import *
B, N, MAX = 16, 26, 8.0
BG = (135, 175, 220)   # sky
COL = [(200, 30, 30), (240, 230, 210)]


def canopy(kind="tri", stripes=8, rot=0, fringe=False, pole=False):
    im = Image.new("RGB", (224, 224), BG); d = ImageDraw.Draw(im); apex, L, R, base = (112, 40), 30, 194, 120
    if kind == "tri":
        for k in range(stripes):
            x0, x1 = L + (R - L) * k / stripes, L + (R - L) * (k + 1) / stripes; d.polygon([apex, (x0, base), (x1, base)], fill=COL[k % 2])
    elif kind == "dome":
        for k in range(stripes):
            d.pieslice([L, 40, R, 200], 180 + 180 * k / stripes, 180 + 180 * (k + 1) / stripes, fill=COL[k % 2])
    elif kind == "rect":
        for k in range(stripes):
            x0, x1 = L + (R - L) * k / stripes, L + (R - L) * (k + 1) / stripes; d.rectangle([x0, 40, x1, base], fill=COL[k % 2])
    elif kind == "plaintri": d.polygon([apex, (L, base), (R, base)], fill=COL[0])
    if fringe:
        for x in range(L, R, 14): d.pieslice([x, base - 7, x + 14, base + 7], 0, 180, fill=COL[(x // 14) % 2])
    if pole: d.rectangle([108, base, 116, 210], fill=(90, 70, 50))
    return np.array(im.rotate(rot, fillcolor=BG))


S = {"striped tri apex-up": canopy(), "striped tri apex-up +fringe+pole": canopy(fringe=True, pole=True), "striped tri apex-DOWN": canopy(rot=180),
     "striped tri rot90": canopy(rot=90), "striped dome (umbrella)": canopy("dome"), "striped dome +pole": canopy("dome", pole=True),
     "striped rectangle": canopy("rect"), "plain red tri apex-up": canopy("plaintri"), "plain red tri apex-down": canopy("plaintri", rot=180),
     "sky only": np.full((224, 224, 3), BG, np.uint8)}
Z = zmaps(B, N, np.stack(list(S.values())))
for k, z in zip(S, Z): print(f"{k:34s} max {z.max():6.2f} ({100*z.max()/MAX:4.0f}% of max)")
