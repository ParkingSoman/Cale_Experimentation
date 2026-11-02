"""Deep view of neurons (essay microscope style, extended).
Usage: python view.py 14,831 16,384 ...   -> research/clean/map/deep/b{b}_n{n}.png and .json
Panels: crops at activation bands (100/75/50/25/10% of max), negative pole (strongest and half-strength), 6 full
images with the neuron's heat map, edit fingerprint (response after edit / before, median over top-20 images)."""
import sys, os, json
import numpy as np, mlx.core as mx
from PIL import Image, ImageDraw, ImageFilter
from convnext import *
import gallery as G

OUT = os.path.join(ROOT, "research", "clean", "map", "deep"); T = 80
EDITS = ["flip", "rot90", "rot180", "gray", "blur1", "blur3", "shuf2", "shuf4", "shuf8", "hue", "invert", "zoomout", "zoomin"]


def edit(im, kind, rng):
    P = Image.fromarray(im)
    if kind == "flip": P = P.transpose(Image.FLIP_LEFT_RIGHT)
    elif kind == "rot90": P = P.rotate(90)
    elif kind == "rot180": P = P.rotate(180)
    elif kind == "gray": P = P.convert("L").convert("RGB")
    elif kind == "blur1": P = P.filter(ImageFilter.GaussianBlur(1))
    elif kind == "blur3": P = P.filter(ImageFilter.GaussianBlur(3))
    elif kind == "zoomout": Q = Image.new("RGB", (224, 224), tuple(int(v) for v in im.reshape(-1, 3).mean(0))); Q.paste(P.resize((112, 112)), (56, 56)); P = Q
    elif kind == "zoomin": P = P.crop((56, 56, 168, 168)).resize((224, 224))
    a = np.array(P)
    if kind == "hue": a = a[..., [1, 2, 0]]
    if kind == "invert": a = 255 - a
    if kind.startswith("shuf"):
        k = int(kind[4:]); s = 224 // k
        t = a.reshape(k, s, k, s, 3).transpose(0, 2, 1, 3, 4).reshape(k * k, s, s, 3)[rng.permutation(k * k)]
        a = t.reshape(k, k, s, s, 3).transpose(0, 2, 1, 3, 4).reshape(224, 224, 3)
    return a


def zmap(b, n, imgs):
    x = G._m.stream(mx.array(norm_img(imgs)), upto=b); return np.array(G._m.blocks[b].fc1(G._m.blocks[b].pre(x))[..., n])


def deep(targets):
    G._init(); m, X, y, IDX = G._m, G._X, G._y, G._idx; names = [s.split(",")[0] for s in class_names()]
    want = {}
    for b, n in targets:
        if n not in want.setdefault(b, []): want[b].append(n)
    S = G.scan(want); rng = np.random.default_rng(0)
    for b, n in targets:
        vm, pm, vn, pn = S[(b, n)]; mx_ = vm.max(); mn_ = vn.min(); rows = []; used = set()
        for f in (1.0, .75, .5, .25, .1):
            order = np.argsort(-vm) if f == 1.0 else np.argsort(np.abs(vm - f * mx_))
            sel = [i for i in order if i not in used][:8]; used.update(sel)
            rows.append((f"+ {int(f*100)}% of max ({f*mx_:.2f})", [(i, pm[i], vm[i]) for i in sel], "darkgreen"))
        rows.append((f"- strongest negative ({mn_:.2f})", [(i, pn[i], vn[i]) for i in np.argsort(vn)[:8]], "darkred"))
        rows.append((f"- half negative ({.5*mn_:.2f})", [(i, pn[i], vn[i]) for i in np.argsort(np.abs(vn - .5 * mn_))[:8]], "darkred"))
        top = np.argsort(-vm)[:20]; base = zmap(b, n, np.asarray(X[IDX[top]])).max((1, 2))
        ed = {}
        for k in EDITS:
            arr = np.stack([edit(np.asarray(X[IDX[i]]), k, rng) for i in top]); ed[k] = float(np.median(zmap(b, n, arr).max((1, 2)) / np.maximum(base, 1e-6)))
        # heat maps on 6 top images
        Z6 = zmap(b, n, np.asarray(X[IDX[top[:6]]]))
        W = T * 8 + 20; Hh = 30 + len(rows) * (T + 24) + 130 + 40
        img = Image.new("RGB", (W, Hh), "white"); dr = ImageDraw.Draw(img)
        dr.text((6, 4), f"block {b} n{n} | on-rate n/a | edits: " + " ".join(f"{k} {v:.2f}" for k, v in ed.items()), fill="black")
        yy = 26
        for lab, cells, col in rows:
            dr.text((6, yy), lab, fill=col); yy += 12
            for j, (i, p, v) in enumerate(cells):
                img.paste(G.crop(i, p, b).resize((T, T)), (6 + j * T, yy)); dr.text((6 + j * T, yy + T - 11), names[y[IDX[i]]][:12], fill="white")
            yy += T + 12
        dr.text((6, yy), "heat: top-6 images (red = positive, blue = negative)", fill="black"); yy += 12
        for j in range(6):
            im = np.asarray(X[IDX[top[j]]]).astype(np.float32); z = Z6[j]; s = max(abs(z).max(), 1e-6)
            hm = np.array(Image.fromarray(((np.clip(z / s, -1, 1) + 1) * 127.5).astype(np.uint8)).resize((224, 224), Image.BILINEAR)).astype(np.float32) / 255 * 2 - 1
            ov = im * .55; ov[..., 0] += 115 * np.clip(hm, 0, 1); ov[..., 2] += 115 * np.clip(-hm, 0, 1)
            img.paste(Image.fromarray(np.clip(ov, 0, 255).astype(np.uint8)).resize((112, 112)), (6 + j * 116, yy))
        fp = os.path.join(OUT, f"b{b}_n{n}.png"); img.save(fp)
        cls = np.bincount(y[IDX[np.argsort(-vm)[:50]]], minlength=1000); clsn = np.bincount(y[IDX[np.argsort(vn)[:50]]], minlength=1000)
        json.dump({"block": b, "neuron": n, "max": float(mx_), "min": float(mn_), "edits": ed,
                   "top50_classes": {names[c]: int(cls[c]) for c in cls.argsort()[::-1][:10]},
                   "neg50_classes": {names[c]: int(clsn[c]) for c in clsn.argsort()[::-1][:10]}}, open(fp.replace(".png", ".json"), "w"), indent=1)
        print(fp, flush=True)


if __name__ == "__main__":
    deep([tuple(int(v) for v in a.split(",")) for a in sys.argv[1:]])
