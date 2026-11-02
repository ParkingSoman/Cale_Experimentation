"""b14 n1326 follow-up (after the wedge test showed oblique straight step edges, not wedge tips, drive it).
PREDICTION: a straight step edge through the centre drives >40% of max at 30-60 deg and 120-150 deg, and <10% at
0 and 90 deg (horizontal / vertical). Also: a soft (blurred 6 px) oblique edge keeps >50% of the sharp response;
a jagged oblique edge (staircase of 16 px steps) drops below half. Value = max over the inner 10x10 cells."""
from lib4 import *
from PIL import Image, ImageDraw, ImageFilter
import math
b, n = 14, 1326; MX = 6.47
def half(theta, stair=0):
    yy, xx = np.mgrid[0:224, 0:224] - 111.5
    if stair: yy, xx = np.floor(yy / stair) * stair, np.floor(xx / stair) * stair
    m = (xx * math.sin(math.radians(theta)) - yy * math.cos(math.radians(theta))) > 0
    return np.where(m[..., None], 200, 70).astype(np.uint8).repeat(3, 2)
angs = list(range(0, 180, 15)); I = np.stack([half(a) for a in angs]); Z = zmap(b, n, I)
print("angle:", angs); print("inner max %:", [int(100 * z[2:12, 2:12].max() / MX) for z in Z])
B = np.stack([np.array(Image.fromarray(half(45)).filter(ImageFilter.GaussianBlur(6))), half(45), half(45, stair=16), half(45, stair=32)])
print("45deg sharp/blur6/stair16/stair32 inner max %:", [int(100 * z[2:12, 2:12].max() / MX) for z in zmap(b, n, B)[[1, 0, 2, 3]]])
