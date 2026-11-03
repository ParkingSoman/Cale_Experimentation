# serves: H-map
"""Step 1 of plan im-looking-at-the-playful-whisper.md. Is the map's density a result of looking across four convolutions?

Definition DC4. For a block: i = relu(a(h)), z = r(i) + s(h). r and a are affine, so a child's change under a fill of its inputs is an exact sum.
   stream child c:  z_c(pos) = z_c^fill(pos) + sum over inner channels j of D_j(pos) + D_skip(pos)
   inner child w:   a_w(pos) = a_w^fill(pos) + sum over stream channels j of D_j(pos)
D_j = (scale of the norm) * sum over the 3 x 3 window of weight * (input_j - mean of input_j). The script checks the sum against the real value.

A child is read at its own peak, on its 64 strongest photos. Lists are chosen on reading photos. Numbers are on sealed photos.
Reported per child:
   n50, n80     how many inputs reach that share of the summed |D_j|: one list per child, one per bundle, one per photo (in-sample, descriptive)
   share_k      E (sum of D_j outside the list)^2 / E z^2, k = 4, 16, 64, as in LADDER 56, for the effect list and a random list (10 seeds)
   signal_k     the same, over E (sum of all D_j)^2
   base         z^fill / z: what is left with every input at its mean
Old measure with its missing control: stage 3 filled at both blocks except k channels, for the old list and a random list (3 seeds).
Uses the CPU: another session is measuring the GPU. Usage: one_layer_check.py [--smoke]"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from net import HERE, STORE, WIDTH, Photos, block_means, block_parts, forward, load, stages, upto  # noqa: E402

smoke = "--smoke" in sys.argv
torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)
READ, SEAL, EDGES = (os.path.join(STORE, x) for x in ("read", "sealed", "edges"))
PH = {"read": Photos("read"), "sealed": Photos("sealed")}
ROOT = {"read": READ, "sealed": SEAL}
m = load().double()
KS = (4, 16, 64)
TOP = 64


def arr(which, name):
    return np.load(os.path.join(ROOT[which], name + ".npy"), mmap_mode="r")


def scale(bn):
    return (bn.weight / torch.sqrt(bn.running_var + bn.eps)).view(-1)


def inner_mean(s, b):
    return torch.as_tensor(np.asarray(arr("read", f"s{s}b{b}_im"), dtype=np.float64).mean(0))


def stream_mean(s, b):
    return torch.as_tensor(np.asarray(arr("read", f"s{s}b{b}_hm"), dtype=np.float64).mean(0))


def parts(which, idx, s):
    """Both blocks of stage s on these photos."""
    x = PH[which].tensor(idx).double()
    with torch.no_grad():
        h = upto(m, x, s)
        lay = stages(m)[s]
        p0 = block_parts(lay[0], h)
        p1 = block_parts(lay[1], p0["h"])
    return p0, p1


def contributions(inp, mean, conv, bn, c, pos):
    """D_j at (photo, pos) of output channel c, from every input channel j. Also the value with all inputs at their means."""
    w = conv.weight[c]                                   # inputs x 3 x 3
    n = inp.shape[1]
    dev = inp - mean.view(1, -1, 1, 1)
    with torch.no_grad():
        out = F.conv2d(dev, w.view(n, 1, 3, 3), padding=1, groups=n) * scale(bn)[c]   # photos x inputs x H x W
        g = out.shape[-1]
        r, q = torch.as_tensor(pos // g), torch.as_tensor(pos % g)
        D = out[torch.arange(out.shape[0]), :, r, q]
        field = mean.view(1, -1, 1, 1).expand_as(inp)
        base = bn(conv(field))[:, c][torch.arange(out.shape[0]), r, q]
    return D, base


def child_inputs(which, s, c, kind):
    """kind 'stream': z_c of block 1 from its inner channels and the skip. kind 'inner': a_w of block 1 from block 0's stream."""
    key = "h" if kind == "stream" else "i"
    v = np.asarray(arr(which, f"s{s}b1_{key}x")[:, c], dtype=np.float64)
    idx = np.sort(np.argsort(-v)[:TOP])
    pos = np.asarray(arr(which, f"s{s}b1_{key}p")[idx, c]).astype(np.int64)
    p0, p1 = parts(which, idx, s)
    bl = stages(m)[s][1]
    g = p1["z"].shape[-1]
    r, q = pos // g, pos % g
    at = np.arange(len(idx))
    if kind == "stream":
        D, base = contributions(p1["inner"], inner_mean(s, 1), bl.conv2, bl.bn2, c, pos)
        skip_mean = stream_mean(s, 0)[c]
        Dskip = p0["h"][:, c][at, r, q] - skip_mean
        true = p1["z"][:, c][at, r, q]
        total = base + D.sum(1) + Dskip + skip_mean
        D = torch.cat([D, Dskip[:, None]], 1)                      # the last input is the skip, as one input
        base = base + skip_mean
    else:
        D, base = contributions(p0["h"], stream_mean(s, 0), bl.conv1, bl.bn1, c, pos)
        true = bl.bn1(bl.conv1(p0["h"]))[:, c][at, r, q]
        total = base + D.sum(1)
    miss = float((total - true).abs().max() / (true.abs().mean() + 1e-9))
    if miss > 1e-6:
        raise SystemExit(f"the sum misses z: s{s}:{c} {kind} relative {miss}")
    return D.numpy(), base.numpy(), true.numpy()


def n_reach(order, weights, share):
    """Inputs in this order that reach this share of the summed effect."""
    cum = np.cumsum(weights[order])
    return int(np.searchsorted(cum, share * cum[-1]) + 1)


def read_child(s, c, kind, rng):
    Dr, _, _ = child_inputs("read", s, c, kind)
    Ds, base, z = child_inputs("sealed", s, c, kind)
    n = Ds.shape[1]
    effect = np.abs(Dr).mean(0)
    order = np.argsort(-effect)
    weights = np.abs(Ds).mean(0)
    res = dict(child=f"s{s}:{c}", kind=kind, inputs=n, z=float(z.mean()), base=float(np.median(base / np.where(np.abs(z) < 1e-9, 1e-9, z))))
    res["n50_list"], res["n80_list"] = n_reach(order, weights, .5), n_reach(order, weights, .8)
    res["n50_best"], res["n80_best"] = n_reach(np.argsort(-weights), weights, .5), n_reach(np.argsort(-weights), weights, .8)
    # one list per bundle: bundles on reading photos, a sealed photo goes to the nearest bundle
    norm = Dr / (np.linalg.norm(Dr, axis=1, keepdims=True) + 1e-9)
    km = KMeans(3, n_init=10, random_state=0).fit(norm)
    lab = km.predict(Ds / (np.linalg.norm(Ds, axis=1, keepdims=True) + 1e-9))
    n50b, n80b = [], []
    for b in range(3):
        w = np.where(lab == b)[0]
        rd = np.where(km.labels_ == b)[0]
        if len(w) == 0 or len(rd) == 0:
            continue
        o = np.argsort(-np.abs(Dr[rd]).mean(0))
        wt = np.abs(Ds[w]).mean(0)
        n50b += [n_reach(o, wt, .5)] * len(w)
        n80b += [n_reach(o, wt, .8)] * len(w)
    res["n50_bundle"], res["n80_bundle"] = float(np.mean(n50b)), float(np.mean(n80b))
    # one list per photo: in-sample, it only describes a photo
    abs_s = np.abs(Ds)
    res["n50_photo"] = float(np.median([n_reach(np.argsort(-a), a, .5) for a in abs_s]))
    res["n80_photo"] = float(np.median([n_reach(np.argsort(-a), a, .8) for a in abs_s]))
    z2, sig2 = float((z ** 2).mean()), float((Ds.sum(1) ** 2).mean())
    for k in KS:
        rest = order[k:]
        e = float((Ds[:, rest].sum(1) ** 2).mean())
        res[f"share{k}"], res[f"signal{k}"] = e / (z2 + 1e-12), e / (sig2 + 1e-12)
        rnd = []
        for _ in range(10):
            keep = rng.choice(n, k, replace=False)
            mask = np.ones(n, bool)
            mask[keep] = False
            rnd.append(float((Ds[:, mask].sum(1) ** 2).mean()))
        res[f"share{k}_rand"], res[f"signal{k}_rand"] = float(np.mean(rnd)) / (z2 + 1e-12), float(np.mean(rnd)) / (sig2 + 1e-12)
    res["top_inputs"] = [int(j) for j in order[:6]]
    return res


MEANS = {s: tuple(v.double() for v in block_means(READ)[s]) for s in (3, 4)}


def old_measure(cs, cc, rng):
    """Stage cs - 1 filled at both blocks except k channels, as score_acts.py. Old list against random lists."""
    ps = cs - 1
    v = np.asarray(arr("sealed", f"s{cs}b1_hx")[:, cc], dtype=np.float64)
    idx = np.sort(np.argsort(-v)[:TOP])
    pos = torch.as_tensor(np.asarray(arr("sealed", f"s{cs}b1_hp")[idx, cc]).astype(np.int64))
    x = PH["sealed"].tensor(idx).double()

    def child(fill):
        with torch.no_grad():
            _, o = forward(m, x, fill)
        return o[(cs, 1)]["z"][:, cc].flatten(1)[torch.arange(len(idx)), pos].numpy()

    base = child(None)
    z2 = float((base ** 2).mean())

    def err(keep):
        rest = torch.tensor(sorted(set(range(WIDTH[ps])) - set(keep)), dtype=torch.long)
        new = child({ps: (rest, MEANS[ps][0][rest], MEANS[ps][1][rest])})
        return float(((new - base) ** 2).mean()) / (z2 + 1e-12)

    par = np.load(os.path.join(EDGES, f"s{cs}_top_par.npy"), mmap_mode="r")[cc]
    o = np.argsort(-np.abs(par))
    res = dict(child=f"s{cs}:{cc}")
    for k in KS:
        res[f"old{k}"] = err(o[:k].tolist())
        res[f"rand{k}"] = float(np.mean([err(rng.choice(WIDTH[ps], k, replace=False).tolist()) for _ in range(3)]))
    return res


def med(rows, key):
    return float(np.median([r[key] for r in rows]))


def summary(name, rows):
    print(f"\n{name}: {len(rows)} children, {rows[0]['inputs']} inputs. z {med(rows, 'z'):.2f}. Left with every input at its mean: {med(rows, 'base'):.2f} of z")
    print(f"  inputs for 50% of the effect: list {med(rows, 'n50_list'):.0f}, bundle {med(rows, 'n50_bundle'):.0f}, photo {med(rows, 'n50_photo'):.0f}, best list on these photos {med(rows, 'n50_best'):.0f}")
    print(f"  inputs for 80% of the effect: list {med(rows, 'n80_list'):.0f}, bundle {med(rows, 'n80_bundle'):.0f}, photo {med(rows, 'n80_photo'):.0f}, best list on these photos {med(rows, 'n80_best'):.0f}")
    for k in KS:
        print(f"  k = {k}: share of z^2 changed {med(rows, f'share{k}'):.3f} (random {med(rows, f'share{k}_rand'):.3f}); share of the change from the mean {med(rows, f'signal{k}'):.3f} (random {med(rows, f'signal{k}_rand'):.3f})")


rng = np.random.default_rng(0)
acts = json.load(open(os.path.join(HERE, "results", "acts_sealed.json")))
slice4 = [int(x["child"].split(":")[1]) for x in acts["need"] if x["child"].startswith("s4:")]
slice3 = [int(x["child"].split(":")[1]) for x in acts["need"] if x["child"].startswith("s3:")]
rand4 = np.random.default_rng(1).choice(512, 30, replace=False).tolist()
if smoke:
    slice4, slice3, rand4 = slice4[:2], slice3[:1], rand4[:1]
out = {}
for name, s, chs in (("s4_slice", 4, slice4), ("s4_random", 4, rand4), ("s3_slice", 3, slice3)):
    out[name] = []
    for c in chs:
        out[name].append(read_child(s, int(c), "stream", rng))
        print(name, c, {k: round(out[name][-1][k], 3) for k in ("n50_list", "n50_bundle", "n50_photo", "share64", "share64_rand")}, flush=True)
    summary(name, out[name])
for name, src, s in (("s4_inner", "s4_slice", 4), ("s3_inner", "s3_slice", 3)):
    seen = []
    for r in out[src]:
        seen += [j for j in r["top_inputs"][:3] if j < WIDTH[s] and j not in seen]
    out[name] = []
    for w in seen[:40 if not smoke else 2]:
        out[name].append(read_child(s, int(w), "inner", rng))
        print(name, w, {k: round(out[name][-1][k], 3) for k in ("n50_list", "n50_bundle", "n50_photo", "share64", "share64_rand")}, flush=True)
    summary(name, out[name])
out["old_measure"] = []
for c in slice4:
    out["old_measure"].append(old_measure(4, int(c), rng))
    print("old", c, {k: round(v, 3) for k, v in out["old_measure"][-1].items() if k != "child"}, flush=True)
print("\nold measure, stage 4, across four convolutions: share of z^2 changed")
for k in KS:
    print(f"  k = {k}: old list {med(out['old_measure'], f'old{k}'):.3f}, random list {med(out['old_measure'], f'rand{k}'):.3f}")
json.dump(out, open(os.path.join(HERE, "results", "one_layer_check" + ("_smoke" if smoke else "") + ".json"), "w"), indent=0)
