"""17 n1129 prediction: a texture of long silky strands (coat feathering, ruffles), not a dog shape.
Tiling a 64 px patch from the peak cell to fill the frame keeps >50% of the original max (median over top-20);
tiling the most negative patch of the same image gives < 0. Tiled patches from curly/smooth-coat breeds
(poodles, Bedlington, coonhound, beagle) give < 0. Same stimuli on C8, C2, C4."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from win_lib import top_images, peak_px, tile, maps
import last_lib as L
T = (17, 1129); tg = [T, (15, 827), (16, 526), (16, 2835)]; tags = ["T", "C8", "C2", "C4"]
imgs, idx = top_images(*T, k=20); Z = maps(imgs, [T])[T]
pos = np.stack([tile(im, *peak_px(z)) for im, z in zip(imgs, Z)]); neg = np.stack([tile(im, *peak_px(z, True)) for im, z in zip(imgs, Z)])
nm = L.names(); cls = [i for i, s in enumerate(nm) if s in ("toy poodle", "miniature poodle", "standard poodle", "Bedlington terrier", "black-and-tan coonhound", "beagle")]
cimgs, _ = top_images(*T, k=20, neg=True, classes=set(cls)); Zc = maps(cimgs, [T])[T]
cur = np.stack([tile(im, *peak_px(z, True)) for im, z in zip(cimgs, Zc)])
A = {k: maps(v, tg) for k, v in (("orig", imgs), ("tile peak", pos), ("tile own-min", neg), ("orig curly/smooth", cimgs), ("tile curly/smooth", cur))}
ref = {t: np.median(A["orig"][t].max((1, 2))) for t in tg}
print("median image max (ratio to target-top orig)")
for k, d in A.items(): print(k.ljust(20) + "  ".join(f"{g} {np.median(d[t].max((1, 2))):6.2f} ({np.median(d[t].max((1, 2)))/ref[t]:+.2f})" for g, t in zip(tags, tg)))
print("per-image ratio tile peak / orig (target):", np.round(A["tile peak"][T].max((1, 2)) / A["orig"][T].max((1, 2)), 2))
print("classes of top-20:", [nm[L.data()[1][i]] for i in idx])
