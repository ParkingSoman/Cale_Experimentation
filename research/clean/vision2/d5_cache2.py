"""D5 extra training data for the reference: stream entering block B for analysis-rank 23-49 images (NOT the sealed scoring
set 20-22) plus mirror-flipped copies of ranks 0-17 and 23-49, with intact logits. -> d5cache/x{B}_more.npz"""
import os, sys, time
import numpy as np, mlx.core as mx
from convnext import *
B = int(sys.argv[1]); t0 = time.time(); m = load_convnext(); X, y = load_images(); ST = m.stage_of
rank = np.zeros(len(y), int)
for c in range(1000): ix = np.where(y == c)[0]; rank[ix] = np.arange(len(ix))
EX = np.where(rank >= 23)[0]; FL = np.r_[np.where(rank < 18)[0], EX]
def run(I, flip):
    xs, ls = [], []
    for a in range(0, len(I), 100):
        im = np.asarray(X[I[a:a + 100]]); im = im[:, :, ::-1] if flip else im
        x = m.stem_norm(m.stem(mx.array(norm_img(np.ascontiguousarray(im)))))
        for b in range(18):
            if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
            if b == B: xs.append(np.array(x).astype(np.float16))
            x = m.blocks[b](x)
        ls.append(np.array(m.head(x)).astype(np.float32))
    return np.concatenate(xs), np.concatenate(ls)
A1, L1 = run(EX, False); A2, L2 = run(FL, True)
np.savez(os.path.join(HF, "imagenet", "d5cache", f"x{B}_more.npz"), XB=np.concatenate([A1, A2]), LB=np.concatenate([L1, L2]))
print(f"extra: {len(EX)} new + {len(FL)} flipped images, {time.time() - t0:.0f}s")
