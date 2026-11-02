"""R10: are territory parents wired as gates? Rules: VISION-LOG.md R10 (parents from label-free k=1000 groups, R9b)."""
import json
import numpy as np, mlx.core as mx, mlx.nn as nn
from sklearn.cluster import KMeans
from convnext import *
from acts import get_acts, next_stream

m = load_convnext(); X, y = load_images(); ana, _ = split(y); amax, _ = get_acts(m)
F = np.load(os.path.join(HF, "imagenet", "feat_tiny_ana20k.npy")); F = F / np.linalg.norm(F, axis=1, keepdims=True)
K = 1000; g = KMeans(K, n_init=1, random_state=0).fit_predict(F); rng = np.random.default_rng(2)
T = {}
for b, a in amax.items():
    cm = np.zeros((K, a.shape[1]), np.float32); np.add.at(cm, g, a); cm /= np.bincount(g, minlength=K)[:, None]
    med = np.median(cm, 0); T[b] = ((cm - med) / (cm.max(0) - med + 1e-9) >= .5).astype(np.float32)


def pct(v, i):            # percentile of entry i among v
    return float((v < v[i]).mean())


out = {}
for b in (13, 15, 16):            # parent block b, child block b+1, same stage
    kid, par = T[b + 1], T[b]; c = (kid.T @ par) / np.maximum(kid.sum(0)[:, None], 1); br = par.sum(0)
    jac = (kid.T @ par) / np.maximum(kid.sum(0)[:, None] + br[None] - kid.T @ par, 1)
    ok = (c >= .8) & (br[None] > kid.sum(0)[:, None]); kids = np.where(ok.any(1))[0]
    kids = rng.choice(kids, min(150, len(kids)), replace=False)
    pb, cb = m.blocks[b], m.blocks[b + 1]
    W2 = np.array(pb.fc2.weight) * np.array(pb.gamma)[:, None]               # (C, H): parent write columns
    ks = np.array(pb.conv_dw.weight).sum((1, 2, 3)) * 0 + np.array(cb.conv_dw.weight).sum((1, 2, 3))   # child's dwconv sums
    U = ks[:, None] * W2; U = U - U.mean(0, keepdims=True); U = U * np.array(cb.norm.weight)[:, None]
    V = np.array(cb.fc1.weight) @ U                                           # (H_child, H_parent) virtual weights
    rows = []
    for n in kids:
        p = int(np.argmax(np.where(ok[n], jac[n], -1)))
        cand = np.where((np.abs(br - br[p]) <= .2 * br[p]) & (np.arange(len(br)) != p))[0]
        if len(cand) == 0: cand = np.argsort(np.abs(br - br[p]))[1:6]
        q = int(rng.choice(cand))
        top = ana[np.sort(np.argsort(-amax[b + 1][:, n])[:20])]
        s = m.stream(mx.array(norm_img(np.asarray(X[top]))), upto=b)
        hb = nn.gelu(pb.fc1(pb.pre(s)))
        f = lambda h: nn.gelu(cb.fc1(cb.pre(s + pb.gamma * pb.fc2(h)))[..., int(n)]).sum()
        a = np.array((hb * mx.grad(f)(hb)).sum((1, 2)))                       # (20, H)
        a = (a / np.abs(a).sum(1, keepdims=True)).mean(0)
        rows.append({"child": int(n), "parent": p, "random": q, "attr_pct": pct(a, p), "attr_pct_random": pct(a, q),
                     "attr_share": float(a[p]), "vw_pct": pct(V[n], p), "vw_pct_random": pct(V[n], q), "vw_pos": bool(V[n, p] > 0)})
    k = lambda key: float(np.median([r[key] for r in rows]))
    out[b] = {"n": len(rows), "attr_pct": k("attr_pct"), "attr_pct_random": k("attr_pct_random"), "vw_pct": k("vw_pct"),
              "vw_pct_random": k("vw_pct_random"), "vw_pos": float(np.mean([r["vw_pos"] for r in rows])),
              "attr_top10pct": float(np.mean([r["attr_pct"] >= .9 for r in rows])), "attr_bottom10pct": float(np.mean([r["attr_pct"] <= .1 for r in rows])),
              "rows": rows}
    o = out[b]
    print(f"block {b} -> {b+1}: n {o['n']} | attribution percentile median {o['attr_pct']:.2f} (random {o['attr_pct_random']:.2f}); "
          f"top 10% {o['attr_top10pct']:.2f}, bottom 10% {o['attr_bottom10pct']:.2f} | virtual weight pct {o['vw_pct']:.2f} (random {o['vw_pct_random']:.2f}), positive {o['vw_pos']:.2f}", flush=True)
json.dump(out, open(os.path.join(RES, "vision-r10-wiring.json"), "w"), indent=1)
