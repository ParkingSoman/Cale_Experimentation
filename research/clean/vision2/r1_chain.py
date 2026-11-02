"""R1: the essay's sports-car chain in ConvNeXt-Tiny. Rules: VISION-LOG.md R1."""
import json, sys
import numpy as np, mlx.core as mx, mlx.nn as nn
from PIL import Image, ImageDraw
from convnext import *
from acts import get_acts, next_stream

CLS = int(sys.argv[1]) if len(sys.argv) > 1 else 817
m = load_convnext(); X, y = load_images(); names = [n.split(",")[0] for n in class_names()]
amax, amin = get_acts(m)
ana, _ = split(y); pos = {int(g): k for k, g in enumerate(ana)}   # amax rows follow `ana`
cand = ana[y[ana] == CLS]; lg = logits(m, np.asarray(X[cand])); idx = [int(i) for i, l in zip(cand, lg) if l.argmax() == CLS]
print(names[CLS], "correct analysis images:", len(idx))


def streams(x):
    s = {}; h = m.stem_norm(m.stem(x))
    for b in range(18):
        if b > 0: h = next_stream(m, h, b - 1)
        s[b] = h; h = m.blocks[b](h)
    return s


def chain(i, first=13):
    x = mx.array(norm_img(np.asarray(X[i:i + 1]))); s = streams(x); out = []
    blk = m.blocks[17]; h17 = nn.gelu(blk.fc1(blk.pre(s[17])))
    f = lambda h: m.head(s[17] + blk.gamma * blk.fc2(h))[0, CLS]
    g = mx.grad(f)(h17); c = np.array((h17 * g).sum((0, 1, 2))); n = int(c.argmax()); out.append((17, n, float(c[n]), float(c[c > 0].sum())))
    for b in range(16, first - 1, -1):
        blk, nb = m.blocks[b], m.blocks[b + 1]; hb = nn.gelu(blk.fc1(blk.pre(s[b])))
        def f(h, b=b, blk=blk, nb=nb, n=n):
            nxt = next_stream(m, s[b] + blk.gamma * blk.fc2(h), b)
            return nn.gelu(nb.fc1(nb.pre(nxt))[..., n]).sum()
        g = mx.grad(f)(hb); c = np.array((hb * g).sum((0, 1, 2))); n = int(c.argmax()); out.append((b, n, float(c[n]), float(c[c > 0].sum())))
    return out[::-1]


def desc(b, n, k=50):
    top = ana[np.argsort(-amax[b][:, n])[:k]]; bot = ana[np.argsort(amin[b][:, n])[:k]]
    ct = np.bincount(y[top], minlength=1000); cb = np.bincount(y[bot], minlength=1000)
    return {"distinct_top50": int((ct > 0).sum()), "top_classes": [names[j] for j in np.argsort(-ct)[:4]],
            "bottom_classes": [names[j] for j in np.argsort(-cb)[:4]], "top": top[:6].tolist(), "bot": bot[:6].tolist()}


res = {"class": names[CLS], "chains": []}
for i in idx[:8]:
    ch = chain(i); res["chains"].append({"image": int(i), "chain": [(b, n, round(c / tot, 3)) for b, n, c, tot in ch]})
    print(i, [(b, n, "share %.2f" % (c / tot)) for b, n, c, tot in ch], flush=True)
first = res["chains"][0]
rows = []
for b, n, _ in first["chain"]:
    d = desc(b, n); rows.append((b, n, d)); print("block", b, "neuron", n, "distinct top50", d["distinct_top50"], "| +", d["top_classes"], "| -", d["bottom_classes"])
res["described"] = [{"block": b, "neuron": n, **{k: v for k, v in d.items() if k not in ("top", "bot")}} for b, n, d in rows]
# render: the car, then per block: 6 positive images, 6 negative images
T = 96; W = T * 13 + 40; Hh = (T + 34) * len(rows) + T + 40
img = Image.new("RGB", (W, Hh), "white"); dr = ImageDraw.Draw(img)
img.paste(Image.fromarray(np.asarray(X[first["image"]])).resize((T, T)), (10, 10)); dr.text((T + 20, 40), f"image {first['image']}: {names[CLS]}", fill="black")
yy = T + 30
for b, n, d in rows:
    dr.text((10, yy - 14), f"block {b} neuron {n}  (distinct classes in top-50: {d['distinct_top50']})   + left 6 | - right 6", fill="black")
    for j, k in enumerate(d["top"]):
        img.paste(Image.fromarray(np.asarray(X[k])).resize((T, T)), (10 + j * T, yy)); dr.text((10 + j * T, yy + T), names[y[k]][:14], fill="darkgreen")
    for j, k in enumerate(d["bot"]):
        img.paste(Image.fromarray(np.asarray(X[k])).resize((T, T)), (30 + (6 + j) * T + T // 2, yy)); dr.text((30 + (6 + j) * T + T // 2, yy + T), names[y[k]][:14], fill="darkred")
    yy += T + 34
fp = os.path.join(HERE, "figs", f"chain_{CLS}.png"); img.save(fp); print("figure", fp)
json.dump(res, open(os.path.join(RES, f"vision-r1-chain-{CLS}.json"), "w"), indent=1)
