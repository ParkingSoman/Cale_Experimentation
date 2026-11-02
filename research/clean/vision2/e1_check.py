"""E1a: edges vs integrated gradients over ALL parents of the previous block, 40 children per link."""
import json
import numpy as np, mlx.core as mx, mlx.nn as nn
from scipy.stats import spearmanr
from convnext import *
from acts import get_acts

m = load_convnext(); X, y = load_images(); ana, _ = split(y); amax, _ = get_acts(m); rng = np.random.default_rng(4)
E = np.load(os.path.join(HF, "imagenet", "e1_edges.npz"))
H = [384]*3 + [768]*3 + [1536]*9 + [3072]*3; OFF = np.concatenate([[0], np.cumsum(H)]); out = {}
for b in (13, 15, 16):
    pb, cb = m.blocks[b], m.blocks[b + 1]; rs = []; tops = []
    for n in rng.choice(H[b + 1], 40, replace=False):
        top = ana[np.sort(np.argsort(-amax[b + 1][:, n])[:20])]
        s = m.stream(mx.array(norm_img(np.asarray(X[top]))), upto=b); h = nn.gelu(pb.fc1(pb.pre(s)))
        child = lambda hh: cb.fc1(cb.pre(s + pb.gamma * pb.fc2(hh)))[..., int(n)].sum()      # pre-activation, as the edge
        gf = mx.grad(child); ig = 0
        for a in (np.arange(16) + .5) / 16: ig = ig + gf(h * a)
        ig = np.array((h * ig / 16).sum((0, 1, 2))); e = E[str(b + 1)][n, OFF[b]:OFF[b + 1]]
        rs.append(spearmanr(e, ig)[0]); k = 20; tops.append(len(set(np.argsort(-e)[:k]) & set(np.argsort(-ig)[:k])) / k)
    out[b] = {"spearman_median": float(np.median(rs)), "top20_overlap_median": float(np.median(tops))}
    print(f"block {b}->{b+1}: Spearman(edge, IG) median {out[b]['spearman_median']:.2f}; top-20 overlap {out[b]['top20_overlap_median']:.2f}", flush=True)
json.dump(out, open(os.path.join(RES, "vision-e1a-check.json"), "w"), indent=1)
