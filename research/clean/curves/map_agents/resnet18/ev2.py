# serves: H-map
"""Microscope, round 2. It shows the network ONE LAYER AT A TIME. It never picks a parent and never groups channels.
Run from research/clean/curves:  .venv/bin/python map_agents/resnet18/ev2.py COMMAND ...

A node is a channel at one layer:
   s4b1:179    stream of stage 4, block 1. Value h = relu(z). z = r(inner) + skip.
   s4b1i:179   inner channel of stage 4, block 1. Value relu(a), where a reads the layer below through one convolution.
   s0:12       the stem. It reads pixels.
Block 0 and block 1 of a stage are different nodes. At stage 4 they share 12 of their top 100 photos.
Photo numbers index the reading photos. These are ImageNet validation crops, not N's training photos. Sealed photos are not reachable from here.

 Look
  deep NODE               rings at 100, 75, 50, 25, 10% of the largest firing, the negative pole at 100 and 50%, 6 whole photos with heat maps. Prints a PNG path. Open it with Read
  edit NODE               firing kept after flip, rot90, rot180, gray, blur, tile shuffles, invert, zoom
  photo NODE RANK         save the node's RANK-th strongest reading photo as a PNG (0 is the strongest), with its firing and the place of the peak. Edit it with PIL, then probe it
  probe NODE FILE...      firing on your own images, as a share of the largest firing on the reading photos. Draw tests with PIL in tests/
  fv NODE [--natural] [--through NODE]   feature visualization. --natural follows the routes real photos take for the first 64 steps
 Connections, each one convolution away
  inputs NODE [--band top|neg] [--k 3]   exact inputs at the node's peak, with signs, overall and per bundle. They sum to the value
  readers NODE            what reads the node one convolution above, on the node's own strong photos, or the logits
  bundles NODE [--k 3]    firings split by exact inputs, with a sheet per bundle
  trace PHOTO [--depth 5] one photo, followed from N's top class one layer at a time
 Test
  acts PARENT CHILD       PARENT must be a direct input. Exact change in CHILD at its peaks when PARENT alone has its mean, against matched controls
 Names
  classes QUERY           class numbers whose name holds QUERY
  photos-of CLASS         the reading photos of a class, with N's top class
--threads N, default 2. Keep calls small: several agents run at once."""
import argparse
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from net import HERE, STORE, STRIDE, WIDTH, Photos, block_parts, class_names, forward, load, stages, stem  # noqa: E402

READ = os.path.join(STORE, "read")
SHEETS = os.path.join(HERE, "sheets2")
CROP = {0: 40, 1: 56, 2: 96, 3: 160, 4: 224}   # large enough to see context; stages 3 and 4 read most of the photo
NODE = re.compile(r"^s(\d)(?:b(\d)(i?))?:(\d+)$")


def parse(text):
    g = NODE.match(text)
    if not g:
        raise SystemExit(f"not a node: {text}. Write s4b1:179, s4b1i:179 or s0:12")
    s, b, i, c = int(g[1]), g[2], g[3], int(g[4])
    if s == 0:
        return dict(s=0, b=0, inner=False, c=c)
    if b is None:
        raise SystemExit(f"{text}: name the block, as in s{s}b1:{c}. Block 0 and block 1 are different nodes")
    return dict(s=s, b=int(b), inner=i == "i", c=c)


def name(n, c=None):
    c = n["c"] if c is None else c
    if n["s"] == 0:
        return f"s0:{c}"
    return f"s{n['s']}b{n['b']}{'i' if n['inner'] else ''}:{c}"


def arr(key):
    return np.load(os.path.join(READ, key + ".npy"), mmap_mode="r")


def pre(n):
    return f"s{n['s']}b{n['b']}_"


def peaks(n, band="top"):
    """(firing per photo, position per photo). Band neg gives the most negative value before the ReLU."""
    if band == "neg":
        if n["s"] == 0:
            raise SystemExit("the stem stores no negative pole")
        v, p = (pre(n) + "an", pre(n) + "ap") if n["inner"] else (pre(n) + "zn", pre(n) + "zp")
        if not os.path.exists(os.path.join(READ, v + ".npy")):
            raise SystemExit("inner negative poles are not in the store yet: run prepare_store2.py")
        return np.asarray(arr(v)[:, n["c"]], dtype=np.float64), np.asarray(arr(p)[:, n["c"]]).astype(np.int64)
    k = "i" if n["inner"] else "h"
    return np.asarray(arr(pre(n) + k + "x")[:, n["c"]], dtype=np.float64), np.asarray(arr(pre(n) + k + "p")[:, n["c"]]).astype(np.int64)


def top_photos(n, band="top", count=64):
    v, p = peaks(n, band)
    o = np.argsort(v if band == "neg" else -v)[:count]
    return np.sort(o), p[np.sort(o)]


def scale(bn):
    return (bn.weight / torch.sqrt(bn.running_var + bn.eps)).view(-1)


def mean_of(key):
    return torch.as_tensor(np.asarray(arr(key), dtype=np.float64).mean(0)).float()


def contributions(inp, mean, conv, bn, c, r, q):
    """D_j at (photo, r, q) of output channel c, from every input channel j, and the value with all inputs at their means."""
    n = inp.shape[1]
    w = conv.weight[c]
    k = w.shape[-1]
    out = F.conv2d(inp - mean.view(1, -1, 1, 1), w.view(n, 1, k, k), stride=conv.stride, padding=conv.padding, groups=n) * scale(bn)[c]
    at = torch.arange(len(r))
    base = bn(conv(mean.view(1, -1, 1, 1).expand_as(inp)))[:, c][at, r, q]
    return out[at, :, r, q], base


def grid(s):
    return 224 // STRIDE[s]


def below(n):
    """(stage, block) of the layer this block reads."""
    if n["s"] == 1 and n["b"] == 0:
        return (0, 0)
    return (n["s"], 0) if n["b"] == 1 else (n["s"] - 1, 1)


def below_name(n, j):
    s, b = below(n)
    return f"s0:{j}" if s == 0 else f"s{s}b{b}:{j}"


def below_mean(n):
    s, b = below(n)
    return mean_of(f"s{s}b{b}_hm")


def exact_inputs(m, x, n, pos):
    """Exact inputs of node n at positions pos. Returns (D photos x inputs, names, base, true value). The sum check stops the run if it misses."""
    with torch.no_grad():
        _, o = forward(m, x)
        s, b, c = n["s"], n["b"], n["c"]
        g = grid(s)
        r, q = torch.as_tensor(pos // g), torch.as_tensor(pos % g)
        at = torch.arange(len(r))
        bl = stages(m)[s][b]
        h_in = o[below(n)]["h"]
        if n["inner"]:
            D, base = contributions(h_in, below_mean(n), bl.conv1, bl.bn1, c, r, q)
            names = [below_name(n, j) for j in range(D.shape[1])]
            true = bl.bn1(bl.conv1(h_in))[:, c][at, r, q]
        else:
            D, base = contributions(o[(s, b)]["inner"], mean_of(pre(n) + "im"), bl.conv2, bl.bn2, c, r, q)
            names = [f"s{s}b{b}i:{j}" for j in range(D.shape[1])]
            true = o[(s, b)]["z"][:, c][at, r, q]
            if bl.downsample is None:
                own = below_mean(n)[c]
                D = torch.cat([D, (h_in[:, c][at, r, q] - own)[:, None]], 1)
                names.append(below_name(n, c) + " (skip)")
                base = base + own
            else:
                Ds, bs = contributions(h_in, below_mean(n), bl.downsample[0], bl.downsample[1], c, r, q)
                D = torch.cat([D, Ds], 1)
                names += [below_name(n, j) + " (skip)" for j in range(Ds.shape[1])]
                base = base + bs
        miss = float((base + D.sum(1) - true).abs().max() / (true.abs().mean() + 1e-6))
        if miss > 2e-3:
            raise SystemExit(f"sum check failed on {name(n)}: relative {miss:.2e}")
    return D.numpy().astype(np.float64), names, base.numpy(), true.numpy()


def net():
    torch.set_num_threads(a.threads)
    return load()


def fmt(pairs):
    return "  ".join(f"{i}:{v:+.2f}" for i, v in pairs)


def tile(idx, pos, values, s, path, title=""):
    from PIL import Image, ImageDraw
    P = Photos("read")
    raw = P.raw(idx)
    g, half, S = grid(s), CROP[s] // 2, 168
    rows = int(np.ceil(len(idx) / 4))
    sheet = Image.new("RGB", (4 * S, rows * (S + 26) + 14), "white")
    d = ImageDraw.Draw(sheet)
    d.text((2, 1), title, fill="black")
    cats, y = class_names(), arr("y")
    for j, i in enumerate(idx):
        r, c = int(pos[j]) // g, int(pos[j]) % g
        cy, cx = int((r + .5) * STRIDE[s]), int((c + .5) * STRIDE[s])
        r0, c0 = min(max(cy - half, 0), 224 - 2 * half), min(max(cx - half, 0), 224 - 2 * half)
        crop = Image.fromarray(raw[j][r0:r0 + 2 * half, c0:c0 + 2 * half]).resize((S, S))
        px, py = (j % 4) * S, 14 + (j // 4) * (S + 26)
        sheet.paste(crop, (px, py))
        d.text((px + 2, py + S + 1), f"#{i} val {values[j]:.1f}", fill="black")
        d.text((px + 2, py + S + 13), cats[int(np.argmax(y[i]))][:26], fill="gray")
    os.makedirs(SHEETS, exist_ok=True)
    sheet.save(path)
    print(path)


def value_field(m, x, n):
    """The node's value before its ReLU, over all positions."""
    with torch.no_grad():
        _, o = forward(m, x)
        if n["s"] == 0:
            return o[(0, 0)]["h"][:, n["c"]]
        if n["inner"]:
            bl = stages(m)[n["s"]][n["b"]]
            return bl.bn1(bl.conv1(o[below(n)]["h"]))[:, n["c"]]
        return o[(n["s"], n["b"])]["z"][:, n["c"]]


def cmd_deep(n):
    """Rings by a share of the largest firing: the channel's raw shape, not percentiles."""
    from PIL import Image, ImageDraw
    m = net()
    P = Photos("read")
    v, p = peaks(n)
    largest = v.max()
    rows = []
    for share in (1.0, .75, .5, .25, .10):
        o = np.argsort(np.abs(v - share * largest))[:8]
        rows.append((f"{int(share * 100)}% of the largest firing ({largest:.1f})", o, p[o], v[o]))
    if n["s"] > 0:
        vn, pn = peaks(n, "neg")
        lo = vn.min()
        rows.append((f"negative pole, strongest ({lo:.1f})", np.argsort(vn)[:8], pn[np.argsort(vn)[:8]], vn[np.argsort(vn)[:8]]))
        o = np.argsort(np.abs(vn - .5 * lo))[:8]
        rows.append(("negative pole, half strength", o, pn[o], vn[o]))
    S, W = 112, 8
    s, half, g = n["s"], CROP[n["s"]] // 2, grid(n["s"])
    top6 = np.argsort(-v)[:6]
    field = value_field(m, P.tensor(top6), n).numpy()
    H = sum(S + 28 for _ in rows) + 2 * 168 + 40
    sheet = Image.new("RGB", (W * S, H), "white")
    d = ImageDraw.Draw(sheet)
    y0 = 2
    d.text((2, y0), f"{name(n)}. On for {(v > .01 * largest).mean():.2f} of the photos. Mean {v.mean():.2f}", fill="black")
    y0 += 14
    cats, ys = class_names(), arr("y")
    for title, idx, pos, vals in rows:
        d.text((2, y0), title, fill="black")
        y0 += 12
        raw = P.raw(idx)
        for j, i in enumerate(idx):
            r, c = int(pos[j]) // g, int(pos[j]) % g
            cy, cx = int((r + .5) * STRIDE[s]), int((c + .5) * STRIDE[s])
            r0, c0 = min(max(cy - half, 0), 224 - 2 * half), min(max(cx - half, 0), 224 - 2 * half)
            sheet.paste(Image.fromarray(raw[j][r0:r0 + 2 * half, c0:c0 + 2 * half]).resize((S, S)), (j * S, y0))
            d.text((j * S + 1, y0 + S), f"{vals[j]:.1f} {cats[int(np.argmax(ys[i]))][:14]}", fill="gray")
        y0 += S + 14
    d.text((2, y0), "whole photos: red is positive, blue is negative, before the ReLU", fill="black")
    y0 += 12
    raw = P.raw(top6)
    big = float(np.abs(field).max()) + 1e-6
    for j in range(6):
        heat = F.interpolate(torch.as_tensor(field[j])[None, None].float(), size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy() / big
        rgb = raw[j].astype(np.float32) * .55 + 110 * .45
        rgb[..., 0] += 140 * np.clip(heat, 0, 1)
        rgb[..., 2] += 140 * np.clip(-heat, 0, 1)
        im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize((W * S // 6, W * S // 6))
        sheet.paste(im, (j * (W * S // 6), y0))
    sheet = sheet.crop((0, 0, W * S, y0 + W * S // 6))
    os.makedirs(SHEETS, exist_ok=True)
    path = os.path.join(SHEETS, f"{name(n).replace(':', '_')}_deep.png")
    sheet.save(path)
    print(path)


def cmd_inputs(n, band, k):
    if n["s"] == 0:
        raise SystemExit("the stem reads pixels")
    from sklearn.cluster import KMeans
    m = net()
    idx, pos = top_photos(n, band, 64)
    D, names, base, true = exact_inputs(m, Photos("read").tensor(idx), n, pos)
    print(f"{name(n)} band {band}, {len(idx)} photos at their peak. Value {true.mean():.2f}. With every input at its mean {base.mean():.2f}. The inputs sum to the difference exactly.")
    mean, weight = D.mean(0), np.abs(D).mean(0)
    o = np.argsort(-weight)
    cum = np.cumsum(weight[o]) / weight.sum()
    print(f"inputs to reach 50%, 80% of the summed effect: {int(np.searchsorted(cum, .5)) + 1}, {int(np.searchsorted(cum, .8)) + 1} of {len(names)}")
    print("largest, mean signed change:", fmt([(names[i], mean[i]) for i in o[:10]]))
    print("most positive:", fmt([(names[i], mean[i]) for i in np.argsort(-mean)[:6]]))
    print("most negative:", fmt([(names[i], mean[i]) for i in np.argsort(mean)[:6]]))
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9))
    cats, y = class_names(), arr("y")
    for b in range(k):
        w = np.where(lab == b)[0]
        me = D[w].mean(0)
        print(f"bundle {b}, {len(w)} photos, N's top class: {[cats[int(np.argmax(y[idx[j]]))][:14] for j in w[:6]]}")
        print("   inputs:", fmt([(names[i], me[i]) for i in np.argsort(-np.abs(me))[:7]]))


def cmd_readers(n):
    cats = class_names()
    s, b, c = n["s"], n["b"], n["c"]
    m = net()
    idx, pos = top_photos(n, "top", 32)
    x = Photos("read").tensor(idx)
    with torch.no_grad():
        _, o = forward(m, x)
        if n["inner"]:                         # read by this block's second convolution
            bl = stages(m)[s][b]
            val, mean, conv, bn, nm = o[(s, b)]["inner"], mean_of(pre(n) + "im"), bl.conv2, bl.bn2, (lambda j: f"s{s}b{b}:{j}")
        elif s == 4 and b == 1:                 # read by the logits
            W = m.fc.weight[:, c] - m.fc.weight[:, c].mean()
            top = torch.argsort(W, descending=True)
            print("strongest + logits:", [(cats[int(i)], round(float(W[i]), 2)) for i in top[:12]])
            print("strongest - logits:", [(cats[int(i)], round(float(W[i]), 2)) for i in top.flip(0)[:12]])
            return
        else:                                   # read by the next block's first convolution
            ts, tb = (s, 1) if (s > 0 and b == 0) else (s + 1, 0)
            bl = stages(m)[ts][tb]
            val = o[(s, b)]["h"]
            mean = mean_of(f"s{s}b{b}_hm")
            conv, bn, nm = bl.conv1, bl.bn1, (lambda j: f"s{ts}b{tb}i:{j}")
        dev = (val[:, c] - mean[c])[:, None]
        w = conv.weight[:, c][:, None]          # outputs x 1 x k x k
        kk = conv.kernel_size[0]
        out = F.conv2d(dev, w.view(-1, 1, kk, kk), stride=conv.stride, padding=conv.padding) * scale(bn).view(1, -1, 1, 1)
        peak = out.abs().flatten(2).max(2).values
        signed = out.flatten(2).gather(2, out.abs().flatten(2).argmax(2, keepdim=True))[..., 0]
    me, wt = signed.mean(0).numpy(), peak.mean(0).numpy()
    o2 = np.argsort(-wt)
    print(f"{name(n)} on its 32 strongest photos: largest change it makes in each reader, mean signed.")
    print("readers:", fmt([(nm(int(j)), me[j]) for j in o2[:10]]))
    print("reads it most +:", fmt([(nm(int(j)), me[j]) for j in np.argsort(-me)[:6]]))
    print("reads it most -:", fmt([(nm(int(j)), me[j]) for j in np.argsort(me)[:6]]))
    if not n["inner"]:
        print("It also passes through the skip of the next block.")


def cmd_bundles(n, k):
    from sklearn.cluster import KMeans
    m = net()
    idx, pos = top_photos(n, "top", 64)
    D, names, _, _ = exact_inputs(m, Photos("read").tensor(idx), n, pos)
    v, _ = peaks(n)
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9))
    for b in range(k):
        w = np.where(lab == b)[0]
        me = D[w].mean(0)
        print(f"bundle {b}: {len(w)} photos. Inputs:", fmt([(names[i], me[i]) for i in np.argsort(-np.abs(me))[:6]]))
        p = os.path.join(SHEETS, f"{name(n).replace(':', '_')}_bundle{b}of{k}.png")
        tile(idx[w[:16]], pos[w[:16]], v[idx[w[:16]]], n["s"], p, f"{name(n)} bundle {b} of {k}")


def cmd_trace(i, depth):
    m = net()
    x = Photos("read").tensor(np.array([i]))
    with torch.no_grad():
        y, o = forward(m, x)
    cats = class_names()
    tt = torch.argsort(y[0], descending=True)
    t, u = int(tt[0]), int(tt[1])
    print(f"#{i} label {cats[int(arr('labels')[i])]}. N's top {cats[t]}, second {cats[u]}, margin {float(y[0, t] - y[0, u]):.2f}")
    h4 = o[(4, 1)]["h"]
    carry = ((m.fc.weight[t] - m.fc.weight[u]) * h4.mean((2, 3))[0]).detach().numpy()
    tops = np.argsort(-carry)[:2]
    print("the last layer carries:", fmt([(f"s4b1:{c}", carry[c]) for c in tops]))

    def follow(n, d):
        if d == 0 or n["s"] == 0:
            return
        f = value_field(m, x, n)[0]
        pos = int(torch.argmax(f.flatten()))
        D, names, base, true = exact_inputs(m, x, n, np.array([pos]))
        o3 = np.argsort(-D[0])[:2]
        print("  " * (depth + 1 - d) + f"{name(n)} value {true[0]:.1f}, with inputs at their means {base[0]:.1f} <-", fmt([(names[j], D[0, j]) for j in o3]))
        for j in o3:
            follow(parse(names[j].replace(" (skip)", "")), d - 1)

    for c in tops:
        follow(dict(s=4, b=1, inner=False, c=int(c)), depth)


def cmd_acts(parent, child):
    m = net()
    idx, pos = top_photos(child, "top", 64)
    D, names, base, true = exact_inputs(m, Photos("read").tensor(idx), child, pos)
    tag = name(parent)
    hits = [j for j, t in enumerate(names) if t.replace(" (skip)", "") == tag]
    if not hits:
        raise SystemExit(f"{tag} is not a direct input of {name(child)}. Use inputs {name(child)}")
    j = hits[0]
    wt = np.abs(D).mean(0)
    rms = np.sqrt((D ** 2).mean(0))
    # matched controls: direct inputs of the child with the closest spread of their own largest firing, outside its top 30 inputs
    spread = []
    for t in names:
        q = parse(t.replace(" (skip)", ""))
        spread.append(float(np.std(arr(pre(q) + ("ix" if q["inner"] else "hx"))[:, q["c"]])))
    spread = np.asarray(spread)
    barred = set(np.argsort(-wt)[:30].tolist()) | {j}
    ctrl = [int(q) for q in np.argsort(np.abs(spread - spread[j])) if int(q) not in barred][:3]
    print(f"{name(child)} at its peaks on its top 64 photos: value {true.mean():.2f}, with every input at its mean {base.mean():.2f}.")
    print(f"{tag} alone at its mean: mean fall {D[:, j].mean():+.3f}, root mean square change {rms[j]:.3f}, rank {int((wt > wt[j]).sum()) + 1} of {len(names)}, share of the summed effect {wt[j] / wt.sum():.3f}")
    print("matched controls, root mean square change:", fmt([(names[q], rms[q]) for q in ctrl]))
    v = np.abs(D[:, j])
    print(f"share of the 64 photos where it moves the child by at least half of its largest effect: {(v > .5 * v.max()).mean():.2f}")


def cmd_edit(n):
    from PIL import Image
    m = net()
    v, _ = peaks(n)
    idx = np.argsort(-v)[:20]
    P = Photos("read")
    raw = P.raw(idx)
    rng = np.random.default_rng(0)

    def shuffle(im, t):
        g = 224 // t
        cells = [im[r:r + g, c:c + g] for r in range(0, g * t, g) for c in range(0, g * t, g)]
        order = rng.permutation(len(cells))
        return np.vstack([np.hstack([cells[order[r * t + c]] for c in range(t)]) for r in range(t)])

    def blur(im, k):
        from PIL import ImageFilter
        return np.asarray(Image.fromarray(im).filter(ImageFilter.GaussianBlur(k)))

    def zoom(im, f):
        P2 = Image.fromarray(im)
        if f > 1:
            d = int(224 * (1 - 1 / f) / 2)
            return np.asarray(P2.crop((d, d, 224 - d, 224 - d)).resize((224, 224)))
        size = int(224 * f)
        out = Image.new("RGB", (224, 224), (128, 128, 128))
        out.paste(P2.resize((size, size)), ((224 - size) // 2, (224 - size) // 2))
        return np.asarray(out)

    edits = dict(flip=lambda i: i[:, ::-1], rot90=lambda i: np.rot90(i), rot180=lambda i: np.rot90(i, 2),
                 gray=lambda i: np.repeat(i.mean(2, keepdims=True).astype(np.uint8), 3, 2), blur1=lambda i: blur(i, 1), blur3=lambda i: blur(i, 3),
                 shuf2=lambda i: shuffle(i, 2), shuf4=lambda i: shuffle(i, 4), shuf8=lambda i: shuffle(i, 8), invert=lambda i: 255 - i,
                 zoomout=lambda i: zoom(i, .6), zoomin=lambda i: zoom(i, 1.5))

    def peak(batch):
        x = P.tensor(None, raw=batch)
        return value_field(m, x, n).flatten(1).max(1).values.numpy()

    base = peak(raw)
    print(f"{name(n)}: top 20 photos, median largest value {np.median(base):.2f}. Share kept, median:")
    print("  ".join(f"{k} {np.median(peak(np.stack([f(i).copy() for i in raw])) / np.where(base == 0, 1, base)):.2f}" for k, f in edits.items()))


def cmd_photo(n, rank, folder):
    from PIL import Image
    v, p = peaks(n)
    i = int(np.argsort(-v)[rank])
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"photo_{i}.png")
    Image.fromarray(Photos("read").raw([i])[0]).save(path)
    g = grid(n["s"])
    r, c = divmod(int(p[i]), g)
    cats, y = class_names(), arr("y")
    print(f"{path}: photo {i}, {name(n)} fires {v[i]:.2f} here, {v[i] / v.max():.2f} of the largest firing on the reading photos. The peak is in cell row {r} column {c} of a {g} by {g} grid, about pixel x {int((c + .5) * 224 / g)} y {int((r + .5) * 224 / g)} of 224. N's top class: {cats[int(np.argmax(y[i]))]}")


def cmd_probe(n, files):
    from PIL import Image
    m = net()
    v, _ = peaks(n)
    below_zero = float(np.abs(peaks(n, "neg")[0]).max())
    P = Photos("read")
    for f in files:
        im = np.asarray(Image.open(f).convert("RGB").resize((224, 224)))[None]
        field = value_field(m, P.tensor(None, raw=im), n).flatten(1)
        val, low = float(field.max()), float(field.min())
        print(f"{f}: largest value {val:.2f}, {val / v.max():.2f} of the largest firing on the reading photos ({v.max():.2f}). Lowest value {low:.2f}, {abs(min(low, 0)) / max(below_zero, 1e-9):.2f} of the largest negative firing")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--band", default="top")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--natural", action="store_true")
    ap.add_argument("--through", default="")
    ap.add_argument("--dir", default=os.path.join(HERE, "tests"))
    a = ap.parse_args()
    cmd = a.command
    if cmd == "deep":
        cmd_deep(parse(a.args[0]))
    elif cmd == "inputs":
        cmd_inputs(parse(a.args[0]), a.band, a.k)
    elif cmd == "readers":
        cmd_readers(parse(a.args[0]))
    elif cmd == "bundles":
        cmd_bundles(parse(a.args[0]), a.k)
    elif cmd == "trace":
        cmd_trace(int(a.args[0]), a.depth)
    elif cmd == "acts":
        cmd_acts(parse(a.args[0]), parse(a.args[1]))
    elif cmd == "edit":
        cmd_edit(parse(a.args[0]))
    elif cmd == "photo":
        cmd_photo(parse(a.args[0]), int(a.args[1]), a.dir)
    elif cmd == "probe":
        cmd_probe(parse(a.args[0]), a.args[1:])
    elif cmd == "fv":
        from fv import show
        torch.set_num_threads(a.threads)
        show(a.args[0], a.natural, a.through)
    elif cmd == "classes":
        print([(i, c) for i, c in enumerate(class_names()) if a.args[0].lower() in c.lower()])
    elif cmd == "photos-of":
        q = int(a.args[0])
        cats, y = class_names(), arr("y")
        print(cats[q], [(int(i), cats[int(np.argmax(y[i]))][:18]) for i in np.where(arr("labels") == q)[0]])
    else:
        print(__doc__)
