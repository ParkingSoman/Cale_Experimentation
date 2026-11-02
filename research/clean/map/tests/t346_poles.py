"""0 n346 (input to 831 and 33): which hues drive it? Flat 12-hue patches (S .7, V .7) + gray, centre z.
Also 3 n92, 3 n169, 12 n548 for the same patches (block 3/12 may need texture; reported anyway).
PREDICTION: 0 n346 positive for purple/magenta (270-330), negative for green (90-150)."""
from lib4 import *
import colorsys
P = []
for hh in range(0, 360, 30):
    r, g, b = colorsys.hsv_to_rgb(hh / 360, .7, .7); P.append(np.full((224, 224, 3), [r * 255, g * 255, b * 255], np.uint8))
P.append(np.full((224, 224, 3), 128, np.uint8)); P = np.stack(P)
for b, n in ((0, 346), (0, 279), (0, 286), (3, 92), (3, 169), (12, 548)):
    Z = zmap(b, n, P); c = Z.shape[1] // 2
    print(f"b{b} n{n} hue 0..330 + gray:", [round(float(z[c, c]), 1) for z in Z])
