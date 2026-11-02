"""S1: exact direct (frozen-norm) contributions of ALL earlier neurons to one child at its strongest spots."""
import sys, json
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *

CB, CN = int(sys.argv[1]), int(sys.argv[2]); NIMG = 50
m = load_convnext(); X, y = load_images(); ana, _ = split(y); names = [n.split(",")[0] for n in class_names()]
IDX = np.concatenate([ana[y[ana] == c][:5] for c in range(1000)]); ST = m.stage_of
H = [m.blocks[b].fc1.weight.shape[0] for b in range(18)]


def frozen_ln(ln, x):
    mu = x.mean(-1, keepdims=True); sg = mx.stop_gradient(mx.sqrt(mx.var(x, -1, keepdims=True) + 1e-6)); return (x - mu) / sg * ln.weight + ln.bias


def run(xb):
    """forward, keeping each block's hidden h_b and the stream before each block."""
    x = m.stem_norm(m.stem(xb)); hs, streams = [], []
    for b in range(CB + 1):
        if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
        streams.append(x); blk = m.blocks[b]
        if b < CB: h = nn.gelu(blk.fc1(blk.pre(x))); hs.append(h); x = x + blk.gamma * blk.fc2(h)
    return hs, streams


def direct(stream_c, pos, delta_fn):
    """child pre-activation at pos, as a function of a write added at some earlier block (direct path)."""
    blk = m.blocks[CB]; d = blk.conv_dw(stream_c)
    return blk.fc1(frozen_ln(blk.norm, d))[:, pos[0], pos[1], CN]


# 1. find the child's top images and strongest spot
vals = []
for i in range(0, len(IDX), 100):
    hs, st = run(mx.array(norm_img(np.asarray(X[IDX[i:i + 100]]))))
    z = m.blocks[CB].fc1(m.blocks[CB].pre(st[CB]))[..., CN]; vals.append(np.array(z.reshape(z.shape[0], -1)))
vals = np.concatenate(vals); top = np.argsort(-vals.max(1))[:NIMG]; spot = vals[top].argmax(1)
R = 56 // 2 ** ST[CB]
print("child top classes:", [names[c] for c in np.bincount(y[IDX[top]], minlength=1000).argsort()[::-1][:8]])

# 2. direct contributions: gradient of z_c(spot) w.r.t. a write added to the stream after block p, through the
#    remaining path with intermediate MLPs skipped and all norms frozen
contrib = {p: np.zeros(H[p]) for p in range(CB)}
for k, i in enumerate(top):
    hs, st = run(mx.array(norm_img(np.asarray(X[IDX[i:i + 1]])))); r, c = divmod(int(spot[k]), R)
    # stream at the end of each stage (right before its downsample) sets the frozen downsample norm
    end = {b - 1: st[b - 1] + m.blocks[b - 1].gamma * m.blocks[b - 1].fc2(hs[b - 1]) for b in range(1, CB + 1) if ST[b] != ST[b - 1]}
    for p in range(CB):
        def f(h, p=p):
            v = m.blocks[p].gamma * m.blocks[p].fc2(h) - m.blocks[p].gamma * m.blocks[p].fc2(mx.zeros_like(h))   # write, no bias
            for b in range(p + 1, CB + 1):
                if ST[b] != ST[b - 1]:
                    ln = m.down_norm[ST[b] - 1]; S = end[b - 1]
                    v = m.down[ST[b] - 1](frozen_ln(ln, S + v)) - m.down[ST[b] - 1](frozen_ln(ln, S))
            return direct(st[CB] + v, (r, c), None).sum() - direct(st[CB], (r, c), None).sum()
        g = mx.grad(f)(hs[p]); contrib[p] += np.array((hs[p] * g).sum((0, 1, 2)))
    if k % 10 == 0: print("image", k, flush=True)
contrib = {p: v / NIMG for p, v in contrib.items()}
allc = np.concatenate([contrib[p] for p in range(CB)]); blk_of = np.concatenate([[p] * H[p] for p in range(CB)])
loc = np.concatenate([np.arange(H[p]) for p in range(CB)])
order = np.argsort(-allc); neg = np.argsort(allc)
tot = np.abs(allc).sum()
res = {"child": [CB, CN], "top_pos": [(int(blk_of[j]), int(loc[j]), float(allc[j] / tot)) for j in order[:15]],
       "top_neg": [(int(blk_of[j]), int(loc[j]), float(allc[j] / tot)) for j in neg[:8]],
       "by_block_pos_share": {p: float(np.maximum(contrib[p], 0).sum() / np.maximum(allc, 0).sum()) for p in range(CB)}}
print("top positive:", res["top_pos"]); print("top negative:", res["top_neg"]); print("positive input by block:", {k: round(v, 3) for k, v in res["by_block_pos_share"].items()})
json.dump(res, open(os.path.join(RES, f"vision-s1-family-{CB}-{CN}.json"), "w"), indent=1)
