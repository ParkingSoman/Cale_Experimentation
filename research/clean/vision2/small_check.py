"""Load small ConvNeXts and check top-1 on 2,000 sealed images."""
import numpy as np, mlx.core as mx
from convnext import *
X, y = load_images(); rank = np.zeros(len(y), int)
for c in range(1000): ix = np.where(y == c)[0]; rank[ix] = np.arange(len(ix))
EV = np.where((rank >= 20) & (rank < 22))[0]
CFG = {"tiny": ((3, 3, 9, 3), (96, 192, 384, 768)), "atto": ((2, 2, 6, 2), (40, 80, 160, 320)), "nano": ((2, 2, 8, 2), (80, 160, 320, 640))}
for n, (d, w) in CFG.items():
    m = load_convnext(n, d, w); lg = logits(m, np.asarray(X[EV])); print(n, "top1", (lg.argmax(1) == y[EV]).mean())
