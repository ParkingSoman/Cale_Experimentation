"""R8: is the R1 chain nested (child territory inside parent territory)? Rules: VISION-LOG.md R8."""
import json
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
from acts import get_acts, next_stream

CLS = 817
m = load_convnext(); X, y = load_images(); names = [n.split(",")[0] for n in class_names()]
amax, _ = get_acts(m); ana, _ = split(y); ya = y[ana]
r1 = json.load(open(os.path.join(RES, f"vision-r1-chain-{CLS}.json")))
rng = np.random.default_rng(0)

# class means of spatial-max z, per block: (1000, hidden)
cm = {b: np.stack([a[ya == c].mean(0) for c in range(1000)]) for b, a in amax.items()}


def territory(b, thr):
    c = cm[b]; med = np.median(c, 0); return (c - med) / (c.max(0) - med + 1e-9) >= thr   # (1000, hidden) bool


def streams(x):
    s = {}; h = m.stem_norm(m.stem(x))
    for b in range(18):
        if b > 0: h = next_stream(m, h, b - 1)
        s[b] = h; h = m.blocks[b](h)
    return s


def contrib(i, b, n):
    """grad x act of block-b neurons on block-(b+1) neuron n's summed GELU activation, image i."""
    s = streams(mx.array(norm_img(np.asarray(X[i:i + 1])))); blk, nb = m.blocks[b], m.blocks[b + 1]
    hb = nn.gelu(blk.fc1(blk.pre(s[b])))
    f = lambda h: nn.gelu(nb.fc1(nb.pre(next_stream(m, s[b] + blk.gamma * blk.fc2(h), b)))[..., n]).sum()
    return np.array((hb * mx.grad(f)(hb)).sum((0, 1, 2)))


out = {}
for thr in (.5, .3):
    T = {b: territory(b, thr) for b in cm}; links = []
    for ch in r1["chains"]:
        for (bp, p, _), (bc, c, _) in zip(ch["chain"][:-1], ch["chain"][1:]):
            kid = T[bc][:, c]; par = T[bp]; nk = kid.sum()
            cont = (par & kid[:, None]).sum(0) / max(nk, 1)              # containment for every block-bp neuron
            br = par.sum(0); jac = (par & kid[:, None]).sum(0) / np.maximum((par | kid[:, None]).sum(0), 1)
            null95 = float(np.quantile(cont[rng.choice(len(cont), 1000)], .95))
            ok = (cont >= .8) & (br > nk); best = int(np.argmax(np.where(ok, jac, -1))) if ok.any() else None
            L = {"image": ch["image"], "parent": [bp, p], "child": [bc, c], "child_breadth": int(nk),
                 "parent_breadth": int(br[p]), "containment": round(float(cont[p]), 3), "null95": round(null95, 3),
                 "nested": bool(cont[p] >= .7 and br[p] > nk and cont[p] > null95),
                 "child_classes": [names[k] for k in np.where(kid)[0][:8]],
                 "parent_classes": [names[k] for k in np.where(par[:, p])[0][:8]]}
            if best is not None:
                L["nested_parent"] = {"neuron": best, "breadth": int(br[best]), "containment": round(float(cont[best]), 3),
                                      "jaccard": round(float(jac[best]), 3), "classes": [names[k] for k in np.where(par[:, best])[0][:8]]}
                if thr == .5:
                    cv = contrib(ch["image"], bp, c); L["nested_parent"]["contrib_rank"] = int((cv > cv[best]).sum()) + 1
                    L["nested_parent"]["contrib_share"] = round(float(cv[best] / cv[cv > 0].sum()), 4)
            links.append(L)
    out[str(thr)] = links
    nn_ = sum(l["nested"] for l in links); ex = sum("nested_parent" in l for l in links)
    print(f"\n== thr {thr}: R1 links nested {nn_}/{len(links)}; a nested parent exists for {ex}/{len(links)}")
    for l in links:
        np_ = l.get("nested_parent")
        print(l["image"], "b%d n%d -> b%d n%d" % (*l["parent"], *l["child"]), "| child %d cls, parent %d cls, contain %.2f (null95 %.2f) %s"
              % (l["child_breadth"], l["parent_breadth"], l["containment"], l["null95"], "NESTED" if l["nested"] else ""),
              "| best nested: " + (f"n{np_['neuron']} {np_['breadth']} cls c{np_['containment']} j{np_['jaccard']} rank {np_.get('contrib_rank')}" if np_ else "none"))
json.dump(out, open(os.path.join(RES, "vision-r8-nesting.json"), "w"), indent=1)
