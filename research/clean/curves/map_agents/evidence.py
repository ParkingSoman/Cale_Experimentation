# serves: H-map
"""Evidence tools for the map agents. They show the network. They never group it: no clustering, no ranking of groups.
Run from research/clean/curves with /opt/homebrew/bin/python3.11 map_agents/evidence.py NET COMMAND ...

  sheet U --band top|high|mid|low|neg [--level L]   contact sheet (PNG path printed): crops around where channel U fires
        top: strongest 16 photos; high: 90-95th percentile; mid: 60-70th; low: just above zero; neg: strongest negative
        pre-activation (the negative pole). Level 0 = last hidden layer; level L = L blocks before it.
  readers U                  what the logits do with U: strongest + and - output weights, with class names, and how much of
                             the logit movement on P goes through U
  feeders U --band top [--level L]   which channels of the layer below feed U on those photos (input x gradient, mean)
  feeders-by-photo U [--level L]     for U's top 24 photos, the top 4 feeders of each photo: firings grouped by route, by eye
  carry I,J,...              on photos I, J (indices into P): the channels that carry N's top class against its second class
  photos-of CLASS            indices into P of the 20 photos of a class number, N's top classes on them
  classes QUERY              class numbers whose name contains QUERY
  ablate U,V,...             KL of N when these channels are set to 0 (readers unchanged), and which classes lose most
  corr U                     channels most correlated with U over P, and over U's strong photos (raw numbers, not groups)
  score U --high a,b --near c,d     U's mean activation on classes a,b against near misses c,d (reviewer's test)

Add --cpu-threads N to limit threads (default 2). Keep calls small: several agents run at once."""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CURVES = os.path.dirname(HERE)
OUT = os.path.join(CURVES, "cache", "map")
SHEETS = os.path.join(HERE, "sheets")
os.makedirs(SHEETS, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("net")
ap.add_argument("command")
ap.add_argument("arg", nargs="?")
ap.add_argument("--band", default="top")
ap.add_argument("--level", type=int, default=0)
ap.add_argument("--high", default="")
ap.add_argument("--near", default="")
ap.add_argument("--cpu-threads", type=int, default=2)
a = ap.parse_args()
meta = np.load(os.path.join(OUT, f"{a.net}_meta.npz"))
photos, labels, W, b = meta["photos"], meta["labels"], meta["W"], meta["b"]
H = np.load(os.path.join(OUT, f"{a.net}_h.npy"), mmap_mode="r")
Z = np.load(os.path.join(OUT, f"{a.net}_z.npy"), mmap_mode="r")
Y = np.load(os.path.join(OUT, f"{a.net}_y.npy"), mmap_mode="r")
n = H.shape[0]


def names():
    import torchvision
    return torchvision.models.ResNet18_Weights.IMAGENET1K_V1.meta["categories"]


def softmax(y):
    y = y - y.max(1, keepdims=True)
    e = np.exp(y)
    return e / e.sum(1, keepdims=True)


def band_photos(u):
    v = np.asarray(H[:, u])
    if a.band == "neg":
        return np.argsort(np.asarray(Z[:, u]))[:16]
    o = np.argsort(-v)
    on = int((v > 1e-6).sum())
    if a.band == "top":
        return o[:16]
    lo, hi = dict(high=(.05, .10), mid=(.30, .40), low=(.90, 1.0))[a.band]
    pick = o[int(lo * on):max(int(hi * on), int(lo * on) + 16)]
    return pick[np.linspace(0, len(pick) - 1, 16).astype(int)]


def model():
    import torch
    import torchvision
    torch.set_num_threads(a.cpu_threads)
    m = getattr(torchvision.models, a.net)(weights="IMAGENET1K_V1").eval()
    for p in m.parameters():
        p.requires_grad_(False)
    blocks = []
    for lay in (m.layer1, m.layer2, m.layer3, m.layer4):
        blocks.extend(list(lay))
    return m, blocks


def prep(idx):
    import torch
    X = np.load(os.path.join(CURVES, "..", "hf", "imagenet", "val_224.npy"), mmap_mode="r")
    sel = photos[np.asarray(idx)]
    raw = np.stack([np.asarray(X[i]) for i in sel])
    x = torch.from_numpy(raw).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return raw, (x - mean) / std


def upto(m, blocks, x, level):
    """(input of block -(level+1), output of it). The output is after its ReLU."""
    h = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    k = len(blocks) - 1 - level
    for bl in blocks[:k]:
        h = bl(h)
    return h, blocks[k]


if a.command == "sheet":
    import torch
    from PIL import Image, ImageDraw
    u = int(a.arg)
    if a.level == 0:
        idx = band_photos(u)
    else:
        # for an inner channel, rank the photos by a first pass on a sample of 1000
        rng = np.random.default_rng(u)
        pool = rng.choice(n, 1000, replace=False)
        m, blocks = model()
        vals = []
        with torch.no_grad():
            for s in range(0, 1000, 50):
                _, x = prep(pool[s:s + 50])
                inp, bl = upto(m, blocks, x, a.level)
                vals.append(bl(inp)[:, u].mean((1, 2)).numpy())
        v = np.concatenate(vals)
        o = np.argsort(-v)
        if a.band == "top":
            idx = pool[o[:16]]
        elif a.band == "neg":
            idx = pool[o[-16:]]
        else:
            lo, hi = dict(high=(.05, .10), mid=(.30, .40), low=(.90, 1.0))[a.band]
            idx = pool[o[int(lo * 1000):int(hi * 1000)]][:16]
    m, blocks = model()
    raw, x = prep(idx)
    with torch.no_grad():
        inp, bl = upto(m, blocks, x, a.level)
        cap = {}
        hk = bl.relu.register_forward_pre_hook(lambda mod, i: cap.__setitem__("z", i[0].clone()))  # relu is in place
        out = bl(inp)
        hk.remove()
        field = cap["z"][:, u].numpy() if a.band == "neg" else out[:, u].numpy()
    S = 168
    sheet = Image.new("RGB", (4 * S, 4 * (S + 26)), "white")
    d = ImageDraw.Draw(sheet)
    cats = names()
    for j in range(len(idx)):
        f = field[j]
        pos = np.unravel_index(np.argmin(f) if a.band == "neg" else np.argmax(f), f.shape)
        r, c = int((pos[0] + .5) * 224 / f.shape[0]), int((pos[1] + .5) * 224 / f.shape[1])
        r0, c0 = min(max(r - 56, 0), 112), min(max(c - 56, 0), 112)
        crop = Image.fromarray(raw[j][r0:r0 + 112, c0:c0 + 112]).resize((S, S))
        px, py = (j % 4) * S, (j // 4) * (S + 26)
        sheet.paste(crop, (px, py))
        top = int(np.argmax(Y[idx[j]]))
        d.text((px + 2, py + S + 1), f"#{idx[j]} val {f[pos]:.1f} mean {f.mean():.1f}", fill="black")
        d.text((px + 2, py + S + 13), cats[top][:26], fill="gray")
    p = os.path.join(SHEETS, f"{a.net}_L{a.level}_u{u}_{a.band}.png")
    sheet.save(p)
    print(p)

elif a.command == "readers":
    u = int(a.arg)
    cats = names()
    w = W[:, u] - W[:, u].mean()
    o = np.argsort(-w)
    print("strongest +:", [(cats[i], round(float(w[i]), 2)) for i in o[:12]])
    print("strongest -:", [(cats[i], round(float(w[i]), 2)) for i in o[-12:][::-1]])
    v = np.asarray(H[:, u])
    print(f"on (h>1% of max) on {float((v > .01 * v.max()).mean()):.3f} of P; mean {v.mean():.3f}; max {v.max():.3f}")
    print("share of total logit movement (std of h x |centred readers|, largest channel = 1):")
    Hstd = np.asarray(H).std(0)
    Wc = W - W.mean(0, keepdims=True)
    s = Hstd * np.linalg.norm(Wc, axis=0)
    print(f"  {s[u] / s.max():.3f}, rank {int((s > s[u]).sum()) + 1} of {len(s)}")

elif a.command in ("feeders", "feeders-by-photo"):
    import torch
    u = int(a.arg)
    m, blocks = model()
    if a.command == "feeders":
        idx = band_photos(u)
    else:
        idx = np.argsort(-np.asarray(H[:, u]))[:24]
    _, x = prep(idx)
    with torch.no_grad():
        inp, bl = upto(m, blocks, x, a.level)
    inp = inp.detach().requires_grad_(True)
    bl(inp)[:, u].mean((1, 2)).sum().backward()
    at = (inp * inp.grad).sum((2, 3)).detach().numpy()  # photos x channels of the layer below
    cats = names()
    if a.command == "feeders":
        mean = at.mean(0)
        o = np.argsort(-mean)
        print("layer below, channel: mean input x gradient on the photos")
        print("feed +:", [(int(i), round(float(mean[i]), 2)) for i in o[:10]])
        print("feed -:", [(int(i), round(float(mean[i]), 2)) for i in o[-6:][::-1]])
        print("sum of all", round(float(mean.sum()), 2))
    else:
        for j in range(len(idx)):
            o = np.argsort(-at[j])[:4]
            print(f"#{idx[j]} {cats[int(np.argmax(Y[idx[j]]))][:22]:22s} h={H[idx[j], u]:.1f} top feeders",
                  [(int(i), round(float(at[j, i]), 1)) for i in o])

elif a.command == "carry":
    cats = names()
    for i in [int(t) for t in a.arg.split(",")]:
        o = np.argsort(-Y[i])
        t, s2 = int(o[0]), int(o[1])
        c = (W[t] - W[s2]) * np.asarray(H[i])
        k = np.argsort(-c)[:8]
        q = softmax(np.asarray(Y[i:i + 1]))[0]
        print(f"#{i} top {cats[t]} {q[t]:.2f}, second {cats[s2]} {q[s2]:.2f}, label {cats[labels[i]]}; margin {Y[i, t] - Y[i, s2]:.2f}")
        print("  channels for top:", [(int(u), round(float(c[u]), 2)) for u in k])
        print("  against:", [(int(u), round(float(c[u]), 2)) for u in np.argsort(c)[:4]])

elif a.command == "photos-of":
    c = int(a.arg)
    idx = np.where(labels == c)[0]
    cats = names()
    print(cats[c], [(int(i), cats[int(np.argmax(Y[i]))][:18]) for i in idx])

elif a.command == "classes":
    q = a.arg.lower()
    print([(i, c) for i, c in enumerate(names()) if q in c.lower()])

elif a.command == "ablate":
    us = [int(t) for t in a.arg.split(",")]
    y = np.asarray(Y, dtype=np.float64)
    h = np.asarray(H, dtype=np.float64)
    y2 = y - h[:, us] @ W[:, us].T
    s = softmax(y)
    ls = np.log(s + 1e-300)
    q = softmax(y2)
    per = (s * (ls - np.log(q + 1e-300))).sum(1)
    print(f"KL {per.mean():.4f} nats; top-1 agreement {float((y.argmax(1) == y2.argmax(1)).mean()):.3f}")
    cats = names()
    byc = np.array([per[labels == c].mean() for c in range(1000)])
    print("classes that lose most:", [(cats[i], round(float(byc[i]), 2)) for i in np.argsort(-byc)[:12]])
    print("share of the KL on the 10% most hit photos:", round(float(np.sort(per)[-n // 10:].sum() / per.sum()), 3))

elif a.command == "corr":
    u = int(a.arg)
    h = np.asarray(H, dtype=np.float64)
    c = np.corrcoef(h.T)[u]
    o = np.argsort(-c)[1:11]
    print("over P:", [(int(i), round(float(c[i]), 2)) for i in o])
    strong = np.argsort(-h[:, u])[:n // 50]
    mu = h[strong].mean(0) / (h.mean(0) + 1e-9)
    o = np.argsort(-mu)[:10]
    print("lift on U's strongest 2% of photos:", [(int(i), round(float(mu[i]), 1)) for i in o])

elif a.command == "score":
    u = int(a.arg)
    v = np.asarray(H[:, u])
    rank = v.argsort().argsort() / (n - 1)
    def group(s):
        return [int(t) for t in s.split(",") if t != ""]
    for name, cl in (("high", group(a.high)), ("near", group(a.near))):
        print(name, [(c, round(float(v[labels == c].mean()), 2), round(float(rank[labels == c].mean()), 2)) for c in cl])
    print("(class, mean activation, mean percentile over P)")
else:
    print(__doc__)
