"""R8c: how common are nested parents, all neurons of blocks 12-17, vs class-shuffled null; plus territory-chain figure."""
import json
import numpy as np
from PIL import Image, ImageDraw
from convnext import *
from acts import get_acts

X, y = load_images(); names = [n.split(",")[0] for n in class_names()]
amax, amin = get_acts(); ana, _ = split(y); ya = y[ana]
cm = {b: np.stack([a[ya == c].mean(0) for c in range(1000)]) for b, a in amax.items()}
T = {}
for b, c in cm.items():
    med = np.median(c, 0); T[b] = (c - med) / (c.max(0) - med + 1e-9) >= .5
rng = np.random.default_rng(0); res = {}
for b in range(12, 18):
    kid = T[b].astype(np.float32); nk = kid.sum(0)
    def best(par):
        inter = kid.T @ par.astype(np.float32); cont = inter / nk[:, None]; br = par.sum(0)
        return ((cont >= .8) & (br[None] > nk[:, None])).any(1)
    real = best(T[b - 1]); shuf = best(T[b - 1][rng.permutation(1000)])
    res[b] = {"child_median_breadth": float(np.median(nk)), "has_nested_parent": float(real.mean()), "null_shuffled": float(shuf.mean())}
    print(f"block {b} <- {b-1}: child median {np.median(nk):.0f} classes | has a nested parent {real.mean():.2f} | class-shuffled null {shuf.mean():.2f}")
json.dump(res, open(os.path.join(RES, "vision-r8c-scale.json"), "w"), indent=1)

chain = [(14, 1072), (15, 1539), (16, 2886), (17, 2283)]
Tt = 96; W = Tt * 13 + 40; Hh = (Tt + 34) * len(chain) + 20
img = Image.new("RGB", (W, Hh), "white"); dr = ImageDraw.Draw(img); yy = 20
for b, n in chain:
    top = ana[np.argsort(-amax[b][:, n])[:200]]; bot = ana[np.argsort(amin[b][:, n])[:6]]
    pick, seen = [], set()                      # 6 top images from 6 different classes, so breadth is visible
    for k in top:
        if y[k] not in seen: seen.add(y[k]); pick.append(k)
        if len(pick) == 6: break
    dr.text((10, yy - 14), f"block {b} neuron {n}: {int(T[b][:, n].sum())} classes in territory   + left (6 different classes) | - right", fill="black")
    for j, k in enumerate(pick):
        img.paste(Image.fromarray(np.asarray(X[k])).resize((Tt, Tt)), (10 + j * Tt, yy)); dr.text((10 + j * Tt, yy + Tt), names[y[k]][:14], fill="darkgreen")
    for j, k in enumerate(bot):
        img.paste(Image.fromarray(np.asarray(X[k])).resize((Tt, Tt)), (30 + (6 + j) * Tt + Tt // 2, yy)); dr.text((30 + (6 + j) * Tt + Tt // 2, yy + Tt), names[y[k]][:14], fill="darkred")
    yy += Tt + 34
fp = os.path.join(HERE, "figs", "territory_chain_817.png"); img.save(fp); print("figure", fp)
