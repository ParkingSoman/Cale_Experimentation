# serves: H-map
"""Evidence for the ResNet-18 map agents. It shows the network. It never picks a parent and never groups channels.
Run from research/clean/curves:  .venv/bin/python map_agents/resnet18/ev.py COMMAND ...
A channel is written s4:179 (stage 1 to 4, channel). s0:12 is the stem. Photo numbers are indices into the reading photos.
These are ImageNet validation crops, not N's training photos. Sealed photos are not reachable from here.

 Look
  sheet CH --band top|high|mid|low|neg [--block 0|1] [--inner]   16 crops around the channel's peak. Prints a PNG path. Open it with Read.
  bundles CH --k 3        firings on the top 64 photos, split by what feeds them. A sheet per bundle. Evidence for one channel only
  edit CH                 largest firing kept on the top 32 photos after grayscale, blur, rotation by 90, flip
 Connections
  parents CH --band top|mid|neg   the channels of the stage below that feed CH, through every route, skip included
  siblings CH --band top|mid|neg  the channels of this stage that feed CH inside the stage; its own channel's entry; correlated channels
  readers CH              what reads CH: the next stage's channels, or the logits for stage 4
  by-photo CH             top 24 photos, top 4 feeders of each: firings grouped by route, by eye
  job CH                  skip share of the variance, and how far the channel keeps one job across the stage
  trace PHOTO             one photo followed from N's top class down through all stages
 Test
  score CH --high a,b --near c,d    percentile of the largest firing on these class numbers
  drop CH,CH,... [--per-class 1]    takes 1 to 2 minutes. Set these channels to their mean through their whole stage: KL of N, classes that lose most
  acts PARENT CHILD       remove PARENT through its stage. How far CHILD falls on CHILD's top 64 photos, against a matched control
  carry I,J,...           on these photos: the stage 4 channels that carry N's top class against its second
 Names
  classes QUERY           class numbers whose name holds QUERY
  photos-of CLASS         indices of the reading photos of a class, with N's top class

Keep calls small. Several agents run at once. --threads N, default 2."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from net import CROP, CURVES, HERE, STORE, STRIDE, WIDTH, parse  # noqa: E402

READ = os.path.join(STORE, "read")
EDGES = os.path.join(STORE, "edges")
SHEETS = os.path.join(HERE, "sheets")
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("command")
ap.add_argument("arg", nargs="?")
ap.add_argument("arg2", nargs="?")
ap.add_argument("--band", default="top")
ap.add_argument("--block", type=int, default=1)
ap.add_argument("--inner", action="store_true")
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--high", default="")
ap.add_argument("--near", default="")
ap.add_argument("--per-class", type=int, default=1)
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
if a.command in ("by-photo", "bundles", "trace", "acts", "drop"):
    a.block = 1  # these commands work on the stage's last value


def arr(name):
    return np.load(os.path.join(READ, name + ".npy"), mmap_mode="r")


def col(name, c):
    return np.asarray(arr(name)[:, c], dtype=np.float64)


def block_of(s):
    return 0 if s == 0 else a.block


def field(s, c, band):
    """(values for ordering, positions, grid side, kind)"""
    b = block_of(s)
    pre = f"s{s}b{b}_"
    if band == "neg":
        if s == 0:
            raise SystemExit("the stem stores no negative pole")
        return -col(pre + "zn", c), np.asarray(arr(pre + "zp")[:, c]), "z"
    if a.inner:
        return col(pre + "ix", c), np.asarray(arr(pre + "ip")[:, c]), "inner"
    return col(pre + "hx", c), np.asarray(arr(pre + "hp")[:, c]), "h"


def band_photos(v, band):
    o = np.argsort(-v)
    if band in ("top", "neg"):
        return o[:16]
    on = int((v > 1e-3).sum())
    lo, hi = dict(high=(.05, .10), mid=(.30, .40), low=(.90, 1.0))[band]
    pick = o[int(lo * on):max(int(hi * on), int(lo * on) + 16)]
    return pick[np.linspace(0, len(pick) - 1, 16).astype(int)]


def names():
    import torchvision
    return torchvision.models.ResNet18_Weights.IMAGENET1K_V1.meta["categories"]


def labels():
    return np.load(os.path.join(READ, "labels.npy"))


def Y():
    return arr("y")


def tile_sheet(idx, pos, values, s, path, title=""):
    from PIL import Image, ImageDraw
    from net import Photos
    P = Photos("read")
    raw = P.raw(idx)
    g = 224 // STRIDE[s]
    S, half = 168, CROP[s] // 2
    rows = int(np.ceil(len(idx) / 4))
    sheet = Image.new("RGB", (4 * S, rows * (S + 26) + 14), "white")
    d = ImageDraw.Draw(sheet)
    d.text((2, 1), title, fill="black")
    cats = names()
    y = Y()
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


def torch_model():
    import torch
    from net import load
    torch.set_num_threads(a.threads)
    return torch, load()


def feeders(torch, m, s, c, idx, pos):
    """Per photo, input x gradient of z at the given position on the input of stage s."""
    from net import Photos, block_parts, stages, upto
    P = Photos("read")
    x = P.tensor(idx)
    with torch.no_grad():
        h_in = upto(m, x, s)
    h_in = h_in.detach().requires_grad_(True)
    lay = stages(m)[s]
    p1 = block_parts(lay[1], block_parts(lay[0], h_in)["h"])
    z = p1["z"][:, c].flatten(1)
    tgt = z[torch.arange(len(idx)), torch.as_tensor(pos.astype(np.int64))].sum()
    g, = torch.autograd.grad(tgt, [h_in])
    return (h_in * g).sum((2, 3)).detach().numpy()


def fmt(pairs):
    return "  ".join(f"{i}:{v:+.2f}" for i, v in pairs)


cmd = a.command
if cmd == "sheet":
    s, c = parse(a.arg)
    v, pos, kind = field(s, c, a.band)
    idx = band_photos(v, a.band)
    vals = -v[idx] if a.band == "neg" else v[idx]
    suffix = "_inner" if a.inner else ""
    p = os.path.join(SHEETS, f"s{s}_{c}_b{block_of(s)}_{a.band}{suffix}.png")
    tile_sheet(idx, pos[idx], vals, s, p, f"s{s}:{c} block {block_of(s)} {kind} band {a.band}")

elif cmd == "parents":
    s, c = parse(a.arg)
    if s == 0:
        raise SystemExit("the stem reads pixels")
    par = np.load(os.path.join(EDGES, f"s{s}_{a.band}_par.npy"), mmap_mode="r")[c]
    o = np.argsort(-par)
    prev = f"s{s - 1}"
    print(f"s{s}:{c} band {a.band}: feeders in {prev}, input x gradient, all routes. Sum of all {par.sum():+.2f}, sum of positives {par[par > 0].sum():+.2f}")
    print("feed +:", fmt([(f"{prev}:{i}", par[i]) for i in o[:10]]))
    print("feed -:", fmt([(f"{prev}:{i}", par[i]) for i in o[::-1][:6]]))
    pos = np.sort(par[par > 0])[::-1]
    if len(pos):
        print(f"share of positive feed in top 1, 5, 20: {pos[0] / pos.sum():.2f} {pos[:5].sum() / pos.sum():.2f} {pos[:20].sum() / pos.sum():.2f}")

elif cmd == "siblings":
    s, c = parse(a.arg)
    lat = np.load(os.path.join(EDGES, f"s{s}_{a.band}_lat.npy"), mmap_mode="r")[c].copy()
    own = lat[c]
    lat[c] = 0
    o = np.argsort(-lat)
    print(f"s{s}:{c} band {a.band}: from block 0 to block 1 of the stage. Own channel {own:+.2f} (the skip plus block 1 reading it). Sum of the others {lat.sum():+.2f}")
    print("feed +:", fmt([(f"s{s}:{i}", lat[i]) for i in o[:8]]))
    print("feed -:", fmt([(f"s{s}:{i}", lat[i]) for i in o[::-1][:6]]))
    hm = np.asarray(arr(f"s{s}b1_hm"), dtype=np.float32)
    u = hm[:, c] - hm[:, c].mean()
    r = (hm - hm.mean(0)).T @ u / (hm.std(0) * u.std() * len(u) + 1e-9)
    r[c] = 0
    o = np.argsort(-r)
    print("most correlated over the photos:", fmt([(f"s{s}:{i}", r[i]) for i in o[:6]]))

elif cmd == "readers":
    s, c = parse(a.arg)
    cats = names()
    if s == 4:
        import torch
        sd = torch.load(os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"), map_location="cpu", weights_only=True)
        W = sd["fc.weight"].detach().numpy()
        w = W[:, c] - W[:, c].mean()
        o = np.argsort(-w)
        print("strongest + logits:", [(cats[i], round(float(w[i]), 2)) for i in o[:12]])
        print("strongest - logits:", [(cats[i], round(float(w[i]), 2)) for i in o[::-1][:12]])
    else:
        nxt = s + 1
        for band in ("top", "neg"):
            par = np.load(os.path.join(EDGES, f"s{nxt}_{band}_par.npy"), mmap_mode="r")[:, c]
            o = np.argsort(-par)
            print(f"band {band} of the reader: s{nxt} channels this feeds + :", fmt([(f"s{nxt}:{i}", par[i]) for i in o[:10]]))
            print(f"band {band} of the reader: s{nxt} channels this feeds - :", fmt([(f"s{nxt}:{i}", par[i]) for i in o[::-1][:6]]))
    v = col(f"s{s}b{block_of(s)}_hm", c)
    print(f"on (mean h > 1% of largest) on {float((v > .01 * v.max()).mean()):.3f} of the photos; mean {v.mean():.3f}; largest {v.max():.3f}")

elif cmd == "by-photo":
    s, c = parse(a.arg)
    torch, m = torch_model()
    v, pos, _ = field(s, c, "top")
    idx = np.argsort(-v)[:24]
    at = feeders(torch, m, s, c, idx, pos[idx])
    cats, y = names(), Y()
    for j, i in enumerate(idx):
        o = np.argsort(-at[j])[:4]
        print(f"#{i} {cats[int(np.argmax(y[i]))][:22]:22s} peak {v[i]:5.1f}  ", fmt([(f"s{s - 1}:{u}", at[j, u]) for u in o]))

elif cmd == "bundles":
    from sklearn.cluster import KMeans
    s, c = parse(a.arg)
    torch, m = torch_model()
    v, pos, _ = field(s, c, "top")
    idx = np.argsort(-v)[:64]
    at = feeders(torch, m, s, c, idx, pos[idx])
    n = at / (np.linalg.norm(at, axis=1, keepdims=True) + 1e-9)
    lab = KMeans(a.k, n_init=10, random_state=0).fit_predict(n)
    cats, y = names(), Y()
    for b in range(a.k):
        w = np.where(lab == b)[0]
        mean = at[w].mean(0)
        o = np.argsort(-mean)[:6]
        top = [cats[int(np.argmax(y[idx[j]]))][:16] for j in w[:8]]
        print(f"bundle {b}: {len(w)} photos; feeders", fmt([(f"s{s - 1}:{u}", mean[u]) for u in o]))
        print("   N's top class on them:", top)
        p = os.path.join(SHEETS, f"s{s}_{c}_bundle{b}of{a.k}.png")
        tile_sheet(idx[w[:16]], pos[idx[w[:16]]], v[idx[w[:16]]], s, p, f"s{s}:{c} bundle {b} of {a.k}")

elif cmd == "edit":
    s, c = parse(a.arg)
    torch, m = torch_model()
    import torch.nn.functional as F
    from net import Photos, forward
    P = Photos("read")
    v, _, _ = field(s, c, "top")
    idx = np.argsort(-v)[:32]
    x = P.tensor(idx)

    def peak(xx):
        with torch.no_grad():
            _, out = forward(m, xx)
        return out[(s, block_of(s))]["h"][:, c].flatten(1).max(1).values

    base = peak(x)
    gray = x.mean(1, keepdim=True).repeat(1, 3, 1, 1)
    k = torch.ones(3, 1, 9, 9) / 81
    edits = dict(grayscale=gray, blur=F.conv2d(x, k, padding=4, groups=3), rot90=torch.rot90(x, 1, (2, 3)), flip=torch.flip(x, (3,)))
    print(f"s{s}:{c}: mean largest firing on its top 32 photos {base.mean():.2f}. Share kept after each edit:")
    for name, xx in edits.items():
        print(f"  {name:10s} {float(peak(xx).mean() / base.mean()):.2f}")

elif cmd == "job":
    s, c = parse(a.arg)
    st = np.load(os.path.join(READ, "skipstats.npz"))
    for b in (0, 1):
        n = st[f"s{s}b{b}_n"]
        r, q = st[f"s{s}b{b}_r"][c] / n, st[f"s{s}b{b}_s"][c] / n
        vr, vs = st[f"s{s}b{b}_rr"][c] / n - r * r, st[f"s{s}b{b}_ss"][c] / n - q * q
        print(f"block {b}: variance of the residual path {vr:.3f}, of the skip {vs:.3f}, skip share {vs / (vr + vs):.2f}")
    h0, h1 = col(f"s{s}b0_hm", c), col(f"s{s}b1_hm", c)
    print(f"correlation over the photos, block 0 against block 1: {np.corrcoef(h0, h1)[0, 1]:.2f}")
    t0, t1 = set(np.argsort(-col(f's{s}b0_hx', c))[:100].tolist()), set(np.argsort(-col(f's{s}b1_hx', c))[:100].tolist())
    print(f"top 100 photos shared by the two blocks: {len(t0 & t1)}")

elif cmd == "trace":
    from net import Photos, block_parts, forward, stages, upto
    torch, m = torch_model()
    i = int(a.arg)
    P = Photos("read")
    x = P.tensor([i])
    with torch.no_grad():
        y, out = forward(m, x)
    cats = names()
    o = torch.argsort(y[0], descending=True)
    t, u = int(o[0]), int(o[1])
    h4 = out[(4, 1)]["h"]
    print(f"#{i} label {cats[int(labels()[i])]}; N's top {cats[t]}, second {cats[u]}, margin {float(y[0, t] - y[0, u]):.2f}")
    carry = ((m.fc.weight[t] - m.fc.weight[u]) * h4.mean((2, 3))[0]).detach().numpy()
    tops = np.argsort(-carry)[:3]
    print("stage 4 carries:", fmt([(f"s4:{c}", carry[c]) for c in tops]))
    frontier = [(4, int(c), 0) for c in tops]
    width = {4: 2, 3: 2, 2: 2}
    while frontier:
        s, c, d = frontier.pop(0)
        if s == 1:
            continue
        hb = out[(s, 1)]["h"][0, c]
        pos = int(torch.argmax(hb.flatten()))
        at = feeders(torch, m, s, c, np.array([i]), np.array([pos]))[0]
        feeds = np.argsort(-at)[:width[s]]
        print("  " * (5 - s) + f"s{s}:{c} (peak {float(hb.max()):.1f}) <-", fmt([(f"s{s - 1}:{q}", at[q]) for q in feeds]))
        frontier += [(s - 1, int(q), d + 1) for q in feeds]

elif cmd == "score":
    s, c = parse(a.arg)
    v = col(f"s{s}b{block_of(s)}_hx", c)
    rank = v.argsort().argsort() / (len(v) - 1)
    lab = labels()
    for name, txt in (("high", a.high), ("near", a.near)):
        cl = [int(t) for t in txt.split(",") if t]
        print(name, [(q, round(float(v[lab == q].mean()), 2), round(float(rank[lab == q].mean()), 2)) for q in cl])
    print("(class, mean largest firing, mean percentile over the reading photos)")

elif cmd in ("drop", "acts"):
    import torch.nn.functional as F
    from net import Photos, forward
    torch, m = torch_model()
    P = Photos("read")

    def fill_of(chs):
        fill = {}
        for ch in chs:
            s, c = parse(ch)
            fill.setdefault(s, []).append(c)
        return {s: (torch.tensor(cs), torch.as_tensor(np.asarray(arr(f"s{s}b{block_of(s)}_hm")[:, cs], dtype=np.float32).mean(0))) for s, cs in fill.items()}

    if cmd == "drop":
        idx = P.per_class(a.per_class)
        chs = a.arg.split(",")
        fill = fill_of(chs)
        kl = []
        stored = Y()
        with torch.no_grad():
            for b in range(0, len(idx), 64):
                sel = idx[b:b + 64]
                y1, _ = forward(m, P.tensor(sel), fill)
                s0 = F.log_softmax(torch.as_tensor(np.asarray(stored[sel], dtype=np.float32)), 1)  # N's logits, from the store
                kl.append((s0.exp() * (s0 - F.log_softmax(y1, 1))).sum(1).numpy())
        kl = np.concatenate(kl)
        lab, cats = labels()[idx], names()
        byc = np.array([kl[lab == c].mean() for c in range(1000)])
        print(f"{len(chs)} channels set to their mean through their stage: KL {kl.mean():.4f} nats on {len(idx)} reading photos")
        print("classes that lose most:", [(cats[i], round(float(byc[i]), 2)) for i in np.argsort(-byc)[:12]])
    else:
        ps, pc = parse(a.arg)
        cs, cc = parse(a.arg2)
        v, pos, _ = field(cs, cc, "top")
        idx = np.argsort(-v)[:64]
        hx = np.asarray(arr(f"s{ps}b{block_of(ps)}_hx"), dtype=np.float32).std(0)
        par = np.load(os.path.join(EDGES, f"s{cs}_top_par.npy"), mmap_mode="r")[cc]
        barred = set(np.argsort(-np.abs(par))[:30].tolist()) | {pc}
        order = np.argsort(np.abs(hx - hx[pc]))
        ctrl = [int(q) for q in order if int(q) not in barred][:3]

        def child(fill):
            vals = []
            with torch.no_grad():
                for b in range(0, len(idx), 32):
                    _, o = forward(m, P.tensor(idx[b:b + 32]), fill)
                    z = o[(cs, block_of(cs))]["z"][:, cc].flatten(1)
                    vals.append(z[torch.arange(z.shape[0]), torch.as_tensor(pos[idx[b:b + 32]].astype(np.int64))].numpy())
            return np.concatenate(vals)

        base = child(None)
        fall = float((base - child(fill_of([a.arg]))).mean())
        falls = [float((base - child(fill_of([f"s{ps}:{q}"]))).mean()) for q in ctrl]
        print(f"{a.arg2} at its own peaks, top 64 photos: mean z {base.mean():.2f}. It falls {fall:+.3f} when {a.arg} is removed.")
        print(f"matched controls (same spread, not in the top 30 feeders) {ctrl}: falls {[round(f, 3) for f in falls]}")

elif cmd == "carry":
    import torch
    sd = torch.load(os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"), map_location="cpu", weights_only=True)
    W = sd["fc.weight"].detach().numpy()
    cats, y, hm, lab = names(), Y(), arr("s4b1_hm"), labels()
    for i in [int(t) for t in a.arg.split(",")]:
        o = np.argsort(-y[i])
        t, u = int(o[0]), int(o[1])
        q = (W[t] - W[u]) * np.asarray(hm[i], dtype=np.float64)
        print(f"#{i} top {cats[t]}, second {cats[u]}, label {cats[lab[i]]}; margin {y[i, t] - y[i, u]:.2f}")
        print("  for top:", fmt([(f"s4:{c}", q[c]) for c in np.argsort(-q)[:8]]))
        print("  against:", fmt([(f"s4:{c}", q[c]) for c in np.argsort(q)[:4]]))

elif cmd == "classes":
    print([(i, c) for i, c in enumerate(names()) if a.arg.lower() in c.lower()])

elif cmd == "photos-of":
    c = int(a.arg)
    cats, y = names(), Y()
    print(cats[c], [(int(i), cats[int(np.argmax(y[i]))][:18]) for i in np.where(labels() == c)[0]])

else:
    print(__doc__)
