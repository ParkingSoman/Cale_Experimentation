"""maps(): fc1 pre-activation maps with optional hidden-unit kill (shared by s6 scripts)."""
import sys, os
import numpy as np, mlx.core as mx, mlx.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "vision2"))
from convnext import norm_img, load_convnext
_m = None


def M():
    global _m
    if _m is None: _m = load_convnext()
    return _m


def maps(imgs, targets, kill=None):
    """{(b,n): (N,R,R)} fc1 pre-activation maps; kill=(b,n) zeroes that hidden unit (post-GELU) in its block."""
    m = M(); ST = m.stage_of; want = {}
    for b, n in targets: want.setdefault(b, []).append(n)
    out = {t: [] for t in targets}
    for i in range(0, len(imgs), 50):
        x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(imgs[i:i + 50])))))
        for b in range(max(want) + 1):
            if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
            blk = m.blocks[b]; z = blk.fc1(blk.pre(x))
            if b in want:
                zz = np.array(z[..., want[b]])
                for j, n in enumerate(want[b]): out[(b, n)].append(zz[..., j])
            if b == max(want): break
            h = nn.gelu(z)
            if kill is not None and kill[0] == b:
                msk = np.ones(h.shape[-1], np.float32); msk[kill[1]] = 0; h = h * mx.array(msk)
            x = x + blk.gamma * blk.fc2(h)
    return {t: np.concatenate(v) for t, v in out.items()}


