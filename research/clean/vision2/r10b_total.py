"""R10b: total effect (integrated gradients + ablation) of territory parents. Rules: VISION-LOG.md R10b."""
import json
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
from acts import get_acts

m = load_convnext(); X, y = load_images(); ana, _ = split(y); amax, _ = get_acts(m)
R10 = json.load(open(os.path.join(RES, "vision-r10-wiring.json"))); out = {}
for bs, d in R10.items():
    b = int(bs); pb, cb = m.blocks[b], m.blocks[b + 1]; rows = []
    for r in d["rows"]:
        n, p, q = r["child"], r["parent"], r["random"]
        top = ana[np.sort(np.argsort(-amax[b + 1][:, n])[:20])]
        s = m.stream(mx.array(norm_img(np.asarray(X[top]))), upto=b); h = nn.gelu(pb.fc1(pb.pre(s)))
        child = lambda hh: nn.gelu(cb.fc1(cb.pre(s + pb.gamma * pb.fc2(hh)))[..., n]).sum((1, 2))   # per image
        gf = mx.grad(lambda hh: child(hh).sum()); ig = 0
        for a in (np.arange(16) + .5) / 16: ig = ig + gf(h * a)
        ig = np.array((h * ig / 16).sum((1, 2)))                                  # (20, H), sums to child(h) - child(0)
        base = np.array(child(h)); scale = np.abs(ig).sum(1, keepdims=True); igm = (ig / scale).mean(0)
        def abl(k):
            mask = np.ones(h.shape[-1], np.float32); mask[k] = 0; return float((base - np.array(child(h * mx.array(mask)))).mean())
        rows.append({"child": n, "ig_pct": float((igm < igm[p]).mean()), "ig_pct_random": float((igm < igm[q]).mean()),
                     "abl": abl(p), "abl_random": abl(q), "child_mean": float(base.mean())})
    med = lambda k: float(np.median([x[k] for x in rows]))
    out[b] = {"ig_pct": med("ig_pct"), "ig_pct_random": med("ig_pct_random"),
              "ig_top10": float(np.mean([x["ig_pct"] >= .9 for x in rows])), "ig_bottom10": float(np.mean([x["ig_pct"] <= .1 for x in rows])),
              "abl_drop_rel": med("abl") / med("child_mean"), "abl_drop_rel_random": med("abl_random") / med("child_mean"),
              "abl_positive": float(np.mean([x["abl"] > 0 for x in rows])), "rows": rows}
    o = out[b]
    print(f"block {b} -> {b+1}: IG pct median {o['ig_pct']:.2f} (random {o['ig_pct_random']:.2f}), top10 {o['ig_top10']:.2f} bottom10 {o['ig_bottom10']:.2f} | "
          f"ablation: child drops in {o['abl_positive']:.2f}; median drop {o['abl_drop_rel']:+.3f} of child (random {o['abl_drop_rel_random']:+.3f})", flush=True)
json.dump(out, open(os.path.join(RES, "vision-r10b-total.json"), "w"), indent=1)
