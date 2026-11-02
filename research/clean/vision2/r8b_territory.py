"""R8b: sign of R1 parents, and a territory (nested) chain. Rules: VISION-LOG.md R8b."""
import json
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
from acts import get_acts, next_stream

CLS = 817
m = load_convnext(); X, y = load_images(); names = [n.split(",")[0] for n in class_names()]
amax, _ = get_acts(m); ana, _ = split(y); ya = y[ana]
r1 = json.load(open(os.path.join(RES, f"vision-r1-chain-{CLS}.json")))
cand = ana[y[ana] == CLS]; lg = logits(m, np.asarray(X[cand])); imgs = [int(i) for i, l in zip(cand, lg) if l.argmax() == CLS]
cm = {b: np.stack([a[ya == c].mean(0) for c in range(1000)]) for b, a in amax.items()}
T = {}
for b, c in cm.items():
    med = np.median(c, 0); T[b] = (c - med) / (c.max(0) - med + 1e-9) >= .5
rng = np.random.default_rng(0)


def streams(x):
    s = {}; h = m.stem_norm(m.stem(x))
    for b in range(18):
        if b > 0: h = next_stream(m, h, b - 1)
        s[b] = h; h = m.blocks[b](h)
    return s


S = {i: streams(mx.array(norm_img(np.asarray(X[i:i + 1])))) for i in imgs}


def contrib(i, b, n):
    s = S[i]; blk, nb = m.blocks[b], m.blocks[b + 1]; hb = nn.gelu(blk.fc1(blk.pre(s[b])))
    f = lambda h: nn.gelu(nb.fc1(nb.pre(next_stream(m, s[b] + blk.gamma * blk.fc2(h), b)))[..., n]).sum()
    c = np.array((hb * mx.grad(f)(hb)).sum((0, 1, 2))); return c, np.array(hb.sum((0, 1, 2)))


out = {"sign": [], "territory": {}}
print("== 1. sign of R1 parents on their image")
on = 0; tot = 0
for ch in r1["chains"]:
    for (bp, p, _), (bc, c, _) in zip(ch["chain"][:-1], ch["chain"][1:]):
        cv, act = contrib(ch["image"], bp, c); tot += 1; on += act[p] > 0
        out["sign"].append({"image": ch["image"], "parent": [bp, p], "child": c, "act": float(act[p]), "contrib": float(cv[p])})
print(f"R1 parents ON (summed GELU > 0): {on}/{tot}")
for e in out["sign"]: print(e["image"], "b%d n%d" % tuple(e["parent"]), "act %+.2f contrib %+.3f" % (e["act"], e["contrib"]))

print("\n== 2-3. territory chains")
for top in (2283, 1594):
    b, n = 17, top; rows = [{"block": 17, "neuron": n, "breadth": int(T[17][:, n].sum()), "classes": [names[k] for k in np.where(T[17][:, n])[0]]}]
    while b - 1 >= 11:
        kid = T[b][:, n]; par = T[b - 1]; nk = kid.sum()
        cont = (par & kid[:, None]).sum(0) / nk; br = par.sum(0)
        jac = (par & kid[:, None]).sum(0) / np.maximum((par | kid[:, None]).sum(0), 1)
        ok = (cont >= .8) & (br > nk)
        if not ok.any(): rows.append({"block": b - 1, "neuron": None}); break
        p = int(np.argmax(np.where(ok, jac, -1)))
        shares = np.stack([contrib(i, b - 1, n)[0] for i in imgs]); shares = shares / np.abs(shares).sum(1, keepdims=True)
        ms = shares.mean(0); rank = int((ms > ms[p]).sum()) + 1; nul = ms[rng.choice(len(ms), 200, replace=False)]
        rows.append({"block": b - 1, "neuron": p, "breadth": int(br[p]), "containment": round(float(cont[p]), 2),
                     "mean_share": round(float(ms[p]), 4), "rank": rank, "null_mean_abs": round(float(np.abs(nul).mean()), 4),
                     "top_contributor": int(ms.argmax()), "top_share": round(float(ms.max()), 4),
                     "classes": [names[k] for k in np.where(par[:, p])[0]]})
        b, n = b - 1, p
    out["territory"][str(top)] = rows
    for r in rows:
        if r.get("neuron") is None: print(f"  block {r['block']}: no nested parent"); continue
        extra = "" if r["block"] == 17 else f" contain {r['containment']} | mean share {r['mean_share']:+.4f} rank {r['rank']} (random |share| {r['null_mean_abs']}, top {r['top_share']:+.4f})"
        print(f"  block {r['block']} n{r['neuron']}: {r['breadth']} classes{extra}\n     {r['classes'][:14]}")
json.dump(out, open(os.path.join(RES, "vision-r8b-territory.json"), "w"), indent=1)
