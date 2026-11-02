"""E1a-diag: edge vs frozen-LN direct contribution vs IG."""
import json
import numpy as np, mlx.core as mx, mlx.nn as nn
from scipy.stats import spearmanr
from convnext import *
from acts import get_acts
m = load_convnext(); X, y = load_images(); ana, _ = split(y); amax, _ = get_acts(m); rng = np.random.default_rng(4)
E = np.load(os.path.join(HF, "imagenet", "e1_edges.npz"))
H = [384]*3 + [768]*3 + [1536]*9 + [3072]*3; OFF = np.concatenate([[0], np.cumsum(H)]); out = {}


def pre_frozen(blk, x):
    d = blk.conv_dw(x); mu = d.mean(-1, keepdims=True); sg = mx.stop_gradient(mx.sqrt(mx.var(d, -1, keepdims=True) + 1e-6))
    return (d - mu) / sg * blk.norm.weight + blk.norm.bias


for b in (13, 15, 16):
    pb, cb = m.blocks[b], m.blocks[b + 1]; r_ef, r_fi, r_ei, r_ew = [], [], [], []
    for n in rng.choice(H[b + 1], 40, replace=False):
        top = ana[np.sort(np.argsort(-amax[b + 1][:, n])[:20])]
        s = m.stream(mx.array(norm_img(np.asarray(X[top]))), upto=b); h = nn.gelu(pb.fc1(pb.pre(s)))
        zfun = lambda hh: cb.fc1(cb.pre(s + pb.gamma * pb.fc2(hh)))[..., int(n)]
        gf = mx.grad(lambda hh: zfun(hh).sum()); ig = 0
        for a in (np.arange(16) + .5) / 16: ig = ig + gf(h * a)
        ig = np.array((h * ig / 16).sum((0, 1, 2)))
        ff = lambda hh: cb.fc1(pre_frozen(cb, s + pb.gamma * pb.fc2(hh)))[..., int(n)]
        fr = np.array((h * mx.grad(lambda hh: ff(hh).sum())(h)).sum((0, 1, 2)))
        z = ff(h); w = mx.maximum(z, 0)                                   # weight positions by the child's on-ness, like the edge
        frw = np.array((h * mx.grad(lambda hh: (ff(hh) * mx.stop_gradient(w)).sum())(h)).sum((0, 1, 2)))
        e = E[str(b + 1)][n, OFF[b]:OFF[b + 1]]
        r_ef.append(spearmanr(e, fr)[0]); r_fi.append(spearmanr(fr, ig)[0]); r_ei.append(spearmanr(e, ig)[0]); r_ew.append(spearmanr(e, frw)[0])
    out[b] = {k: float(np.nanmedian(v)) for k, v in (("edge_frozen", r_ef), ("frozen_ig", r_fi), ("edge_ig", r_ei), ("edge_frozen_onweighted", r_ew))}
    print(f"block {b}->{b+1}: edge~frozen {out[b]['edge_frozen']:.2f} | edge~frozen(on-weighted) {out[b]['edge_frozen_onweighted']:.2f} | frozen~IG {out[b]['frozen_ig']:.2f} | edge~IG {out[b]['edge_ig']:.2f}", flush=True)
json.dump(out, open(os.path.join(RES, "vision-e1a-diag.json"), "w"), indent=1)
