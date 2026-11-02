"""R10c: signed IG share, nested parent vs activation-matched random parent. Rules: VISION-LOG.md R10c."""
import json
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
from acts import get_acts

m = load_convnext(); X, y = load_images(); ana, _ = split(y); amax, _ = get_acts(m); rng = np.random.default_rng(3)
R10 = json.load(open(os.path.join(RES, "vision-r10-wiring.json"))); out = {}
for bs, d in R10.items():
    b = int(bs); pb, cb = m.blocks[b], m.blocks[b + 1]; rows = []
    for r in d["rows"]:
        n, p = r["child"], r["parent"]
        top = ana[np.sort(np.argsort(-amax[b + 1][:, n])[:20])]
        s = m.stream(mx.array(norm_img(np.asarray(X[top]))), upto=b); h = nn.gelu(pb.fc1(pb.pre(s)))
        act = np.array(h.mean((0, 1, 2)))
        cand = np.where((np.abs(act - act[p]) <= .2 * abs(act[p])) & (np.arange(len(act)) != p))[0]
        if len(cand) == 0: cand = np.argsort(np.abs(act - act[p]))[1:6]
        qs = rng.choice(cand, min(10, len(cand)), replace=False)
        child = lambda hh: nn.gelu(cb.fc1(cb.pre(s + pb.gamma * pb.fc2(hh)))[..., n]).sum()
        gf = mx.grad(child); ig = 0
        for a in (np.arange(16) + .5) / 16: ig = ig + gf(h * a)
        ig = np.array((h * ig / 16).sum((0, 1, 2)))
        rows.append({"child": n, "parent": p, "ig": float(ig[p]), "ig_random": [float(ig[q]) for q in qs],
                     "act": float(act[p]), "act_random": [float(act[q]) for q in qs]})
    pos = float(np.mean([x["ig"] > 0 for x in rows])); posr = float(np.mean([v > 0 for x in rows for v in x["ig_random"]]))
    mag = float(np.median([abs(x["ig"]) for x in rows])); magr = float(np.median([abs(v) for x in rows for v in x["ig_random"]]))
    out[b] = {"nested_positive": pos, "random_positive": posr, "nested_abs_median": mag, "random_abs_median": magr, "rows": rows}
    print(f"block {b} -> {b+1}: IG positive: nested {pos:.2f} vs activation-matched random {posr:.2f} | median |IG| nested {mag:.3f} vs random {magr:.3f}", flush=True)
json.dump(out, open(os.path.join(RES, "vision-r10c-signed.json"), "w"), indent=1)
