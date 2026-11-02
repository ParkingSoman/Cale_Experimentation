"""Essay check for 17 n2283: do the shared car+dog neurons (15 n72, 16 n2117, 16 n2521) act as a gate in front of the
car-vs-dog split? (a) inside vs outside each candidate's on-region: 2283's max on car images and min on dog images.
(b) ablation: zero the candidate's GELU output everywhere, measure 2283 max on car images / min on dog images."""
import numpy as np, mlx.core as mx, mlx.nn as nn
from last_lib import *
ya = labels_ana(); X, y, ana = data()
CARS = [407, 436, 468, 511, 609, 627, 656, 661, 717, 751, 817, 581, 479]; DOGS = list(range(151, 269))
car, dog = np.isin(ya, CARS), np.isin(ya, DOGS); v, w = amax(17, 2283), amin(17, 2283)
CANDS = [(15, 72), (16, 2117), (16, 2521), (14, 28)]
RUN_A = True
if RUN_A: print("(a) 2283 inside vs outside candidate on-region (on = max > .5 x 99.9th pct)")
for b, n in (CANDS if RUN_A else []):
    a = amax(b, n); on = a > .5 * np.percentile(a, 99.9)
    print(f"  {b} n{n}: cars on {(car&on).sum():3d}/{car.sum()}  2283 max on cars in/out {v[car&on].mean():5.2f}/{v[car&~on].mean():5.2f} | "
          f"dogs on {(dog&on).sum():4d}/{dog.sum()}  2283 min on dogs in/out {w[dog&on].mean():6.2f}/{w[dog&~on].mean():6.2f}  2283 max on dogs in/out {v[dog&on].mean():5.2f}/{v[dog&~on].mean():5.2f}")
m = model(); rng = np.random.default_rng(0)
ci = ana[car]; di = ana[dog][rng.permutation(dog.sum())[:260]]


def z2283(imgs, abl=None):
    out = []
    for i in range(0, len(imgs), 50):
        x = m.stream(mx.array(norm_img(np.asarray(imgs[i:i + 50]))), upto=abl[0] if abl else 17)
        if abl:
            for q in range(abl[0], 17):
                if q > abl[0] and m.stage_of[q] != m.stage_of[q - 1]: x = m.down[m.stage_of[q] - 1](m.down_norm[m.stage_of[q] - 1](x))
                blk = m.blocks[q]; h = nn.gelu(blk.fc1(blk.pre(x)))
                if q == abl[0]: h = h * mx.array(np.where(np.arange(h.shape[-1]) == abl[1], 0., 1.).astype(np.float32))
                x = x + blk.gamma * blk.fc2(h)
        out.append(np.array(m.blocks[17].fc1(m.blocks[17].pre(x))[..., 2283]))
    return np.concatenate(out)


Ic = np.stack([np.asarray(X[i]) for i in ci]); Id = np.stack([np.asarray(X[i]) for i in di])
bc, bd = z2283(Ic), z2283(Id)
print("(b) ablation (zero candidate's output). baseline 2283: cars mean max %.2f, dogs mean min %.2f, dogs mean max %.2f" % (bc.max((1, 2)).mean(), bd.min((1, 2)).mean(), bd.max((1, 2)).mean()))
for b, n in CANDS + [(16, 2886), (15, 1331)]:
    ac, ad = z2283(Ic, (b, n)), z2283(Id, (b, n))
    print(f"  zero {b} n{n}: cars max change {(ac.max((1,2)) - bc.max((1,2))).mean():+.3f} | dogs min change {(ad.min((1,2)) - bd.min((1,2))).mean():+.3f}, dogs max change {(ad.max((1,2)) - bd.max((1,2))).mean():+.3f}")
