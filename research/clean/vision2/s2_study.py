"""S2: batched hand-study material for 21 neurons. Rules: VISION-LOG.md S2."""
import json, time
import numpy as np, mlx.core as mx, mlx.nn as nn
from PIL import Image, ImageDraw, ImageFilter
from convnext import *
import gallery as G

t0 = time.time(); ST = None
Mm = np.load(os.path.join(HF, "imagenet", "m3_matrices.npz"))
rng = np.random.default_rng(0); BANDS = {"early": range(1, 6), "middle": range(6, 12), "latemid": range(12, 15), "last": range(15, 18)}
anchors = []
for band, bl in BANDS.items():
    for j, b in enumerate(rng.choice(list(bl), 5, replace=len(bl) < 5)):
        rate = Mm[f"{b}_rate"]; n = int(rng.choice(np.where((rate > .001) & (rate < .5))[0])); anchors.append((band, int(b), n, j == 4))
anchors.append(("last", 17, 2283, False))
print("anchors:", anchors, flush=True)
G._init(); m, X, y, IDX = G._m, G._X, G._y, G._idx; ST = m.stage_of
H = [m.blocks[b].fc1.weight.shape[0] for b in range(18)]; names = [n.split(",")[0] for n in class_names()]
want = {}
for _, b, n, _ in anchors: want.setdefault(b, []).append(n)
S = G.scan(want); print("scan", round(time.time() - t0), "s", flush=True)


def fln(ln, x):
    mu = x.mean(-1, keepdims=True); sg = mx.stop_gradient(mx.sqrt(mx.var(x, -1, keepdims=True) + 1e-6)); return (x - mu) / sg * ln.weight + ln.bias


def run(xb, upto):
    x = m.stem_norm(m.stem(xb)); hs, st, end = [], [], {}
    for b in range(upto + 1):
        if b > 0 and ST[b] != ST[b - 1]: end[b - 1] = x; x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
        st.append(x); blk = m.blocks[b]; h = nn.gelu(blk.fc1(blk.pre(x))); hs.append(h); x = x + blk.gamma * blk.fc2(h)
    return hs, st, end, x


def carry(v, s0, s1, end, b_of_stage_end):
    for t in range(s0, s1):
        e = end[b_of_stage_end[t]]; ln, dn = m.down_norm[t], m.down[t]
        v = dn(fln(ln, e + v)) - dn(fln(ln, e))
    return v


SEND = {0: 2, 1: 5, 2: 14}                                  # last block of stages 0-2
cards = []
for band, b, n, held in anchors:
    vm, pm, _, _ = S[(b, n)]; top = np.argsort(-vm)[:20]; xb = mx.array(norm_img(np.asarray(X[IDX[top]])))
    hs, st, end, _ = run(xb, 17); R = 56 // 2 ** ST[b]; rr, cc = pm[top] // R, pm[top] % R
    # upstream: one gradient w.r.t. all earlier hiddens, direct path, frozen norms
    def f(hl):
        tot = None
        for s in range(ST[b] + 1):
            ps = [p for p in range(b) if ST[p] == s]
            if not ps: continue
            v = sum(m.blocks[p].gamma * (m.blocks[p].fc2(hl[p]) - m.blocks[p].fc2(mx.stop_gradient(hl[p]))) for p in ps)
            v = carry(v, s, ST[b], end, SEND); tot = v if tot is None else tot + v
        blk = m.blocks[b]; z = blk.fc1(fln(blk.norm, blk.conv_dw(st[b] + tot)))[..., n]
        return z[mx.arange(len(top)), mx.array(rr), mx.array(cc)].sum()
    up = []
    if b > 0:
        g = mx.grad(f)(hs[:b]); con = np.concatenate([np.array((hs[p] * g[p]).sum((0, 1, 2))) for p in range(b)]) / len(top)
        bo = np.concatenate([[p] * H[p] for p in range(b)]); lo = np.concatenate([np.arange(H[p]) for p in range(b)]); tt = np.abs(con).sum()
        up = [(int(bo[j]), int(lo[j]), float(con[j] / tt)) for j in np.argsort(-con)[:5]] + [(int(bo[j]), int(lo[j]), float(con[j] / tt)) for j in np.argsort(con)[:2]]
    # downstream: carry this neuron's write forward, reading strength where each reader is on
    down = []
    if b < 17:
        mask = np.zeros(H[b], np.float32); mask[n] = 1; v = m.blocks[b].gamma * m.blocks[b].fc2(hs[b] * mx.array(mask)) - m.blocks[b].gamma * m.blocks[b].fc2(mx.zeros_like(hs[b]))
        best = []
        for q in range(b + 1, 18):
            if ST[q] != ST[q - 1]: v = carry(v, ST[q - 1], ST[q], end, SEND)
            blk = m.blocks[q]; d0 = blk.conv_dw(st[q]); zq = blk.fc1(blk.norm(d0))
            dz = blk.fc1(fln(blk.norm, d0 + blk.conv_dw(v) - blk.conv_dw.bias)) - blk.fc1(fln(blk.norm, d0))
            sc = np.array((dz * mx.maximum(zq, 0)).sum((0, 1, 2))) / np.maximum(np.array(mx.maximum(zq, 0).sum((0, 1, 2))), 1e-6)
            for k in np.argsort(-sc)[:3]: best.append((float(sc[k]), q, int(k)))
        down = [(q, k, s) for s, q, k in sorted(best, reverse=True)[:3]]
    # edit tests
    imgs = np.asarray(X[IDX[top]]).copy()
    def edit(kind):
        out = []
        for im in imgs:
            P = Image.fromarray(im)
            if kind == "flip": P = P.transpose(Image.FLIP_LEFT_RIGHT)
            if kind == "rot90": P = P.rotate(90)
            if kind == "gray": P = P.convert("L").convert("RGB")
            if kind == "blur": P = P.filter(ImageFilter.GaussianBlur(3))
            a = np.array(P)
            if kind == "shuffle":
                t = a.reshape(4, 56, 4, 56, 3).transpose(0, 2, 1, 3, 4).reshape(16, 56, 56, 3)[rng.permutation(16)]
                a = t.reshape(4, 4, 56, 56, 3).transpose(0, 2, 1, 3, 4).reshape(224, 224, 3)
            out.append(a)
        return np.stack(out)
    def resp(arr):
        x = m.stream(mx.array(norm_img(arr)), upto=b); return np.array(m.blocks[b].fc1(m.blocks[b].pre(x))[..., n].max((1, 2)))
    base = resp(imgs); ed = {k: float(np.median(resp(edit(k)) / np.maximum(base, 1e-6))) for k in ("flip", "rot90", "gray", "blur", "shuffle")}
    cls = np.bincount(y[IDX[np.argsort(-vm)[:50]]], minlength=1000)
    cards.append({"band": band, "block": b, "neuron": n, "held_out": held, "up": up, "down": down, "edits": ed,
                  "top50_classes": [names[c] for c in cls.argsort()[::-1][:6]], "distinct_top50": int((cls > 0).sum())})
    print(band, b, n, "held" if held else "", "| up", up[:3], "| down", down[:2], "| edits", {k: round(v, 2) for k, v in ed.items()}, flush=True)
json.dump(cards, open(os.path.join(RES, "vision-s2-cards.json"), "w"), indent=1)
print("study done", round(time.time() - t0), "s", flush=True)
# galleries: per band, each anchor + top-2 upstream + top negative upstream + top-2 readers
for band in BANDS:
    rows = []
    for c in [c for c in cards if c["band"] == band]:
        tag = " (HELD OUT)" if c["held_out"] else ""
        rows.append((c["block"], c["neuron"], f"ANCHOR{tag} edits {c['edits']}"))
        for b2, n2, sh in c["up"][:2]: rows.append((b2, n2, f"   input + (share {sh:+.3f})"))
        if c["up"]: b2, n2, sh = c["up"][5]; rows.append((b2, n2, f"   input - (share {sh:+.3f})"))
        for q, k, s in c["down"][:2]: rows.append((q, k, f"   reader (strength {s:+.2f})"))
    print(G.gallery(rows, f"s2_{band}.png", f"S2 {band}: anchor, its top inputs (+,+,-), its top readers", k=6), flush=True)
print("all done", round(time.time() - t0), "s")
