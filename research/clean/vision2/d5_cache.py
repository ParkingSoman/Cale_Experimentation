"""D5 cache: the stream entering block B for all 20k analysis images (rank < 20) and the 3k sealed N11 images (rank
20-22), plus the intact model's logits. hf/imagenet/d5cache/x{B}.npz (float16). usage: python d5_cache.py B"""
import os, sys, time
import numpy as np, mlx.core as mx
from convnext import *
B = int(sys.argv[1]); t0 = time.time(); m = load_convnext(); X, y = load_images(); ST = m.stage_of
rank = np.zeros(len(y), int)
for c in range(1000): ix = np.where(y == c)[0]; rank[ix] = np.arange(len(ix))
ANA = np.where(rank < 20)[0]; EV = np.where((rank >= 20) & (rank < 23))[0]
def run(I):
    xs, ls = [], []
    for a in range(0, len(I), 100):
        x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(X[I[a:a + 100]])))))
        for b in range(18):
            if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
            if b == B: xs.append(np.array(x).astype(np.float16))
            x = m.blocks[b](x)
        ls.append(np.array(m.head(x)))
    return np.concatenate(xs), np.concatenate(ls)
XA, LA = run(ANA); XE, LE = run(EV); D = os.path.join(HF, "imagenet", "d5cache"); os.makedirs(D, exist_ok=True)
np.savez(os.path.join(D, f"x{B}.npz"), XA=XA, LA=LA, XE=XE, LE=LE, ana=ANA, ev=EV, yA=y[ANA], yE=y[EV])
print(f"cached block-{B} inputs: analysis {XA.shape}, sealed {XE.shape}, {time.time() - t0:.0f}s")
