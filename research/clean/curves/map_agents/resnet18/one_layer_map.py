# serves: H-map
"""How clear is each layer of ResNet-18, one convolution at a time? And does position inside the 3 x 3 window help where a layer is less clear?
15 random channels per layer, seed 2. Lists chosen on reading photos, numbers on sealed photos, at each node's own peak on its 64 strongest photos.
   signal_k  E (sum of the inputs outside the list)^2 / E (sum of all inputs)^2      share_k  the same over E value^2     n50  inputs for half of the summed |effect|
Position: inputs are (channel, offset in the 3 x 3 window), for inner nodes at s3 and s4. CPU only. Usage: one_layer_map.py [--smoke]"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ev2 import below, below_mean, exact_inputs, grid, parse, scale  # noqa: E402
from net import HERE, STORE, WIDTH, Photos, forward, load, stages  # noqa: E402

smoke = "--smoke" in sys.argv
torch.set_num_threads(4)
m = load()
PH = {w: Photos(w) for w in ("read", "sealed")}
KS = (4, 16, 64)


def top(which, n):
    pre = f"s{n['s']}b{n['b']}_{'i' if n['inner'] else 'h'}"
    root = os.path.join(STORE, which)
    v = np.asarray(np.load(os.path.join(root, pre + "x.npy"), mmap_mode="r")[:, n["c"]], dtype=np.float64)
    idx = np.sort(np.argsort(-v)[:64])
    return idx, np.asarray(np.load(os.path.join(root, pre + "p.npy"), mmap_mode="r")[idx, n["c"]]).astype(np.int64)


def by_window(which, n, idx, pos):
    """D at (photo, channel x offset) for an inner node."""
    x = PH[which].tensor(idx)
    with torch.no_grad():
        _, o = forward(m, x)
        bl = stages(m)[n["s"]][n["b"]]
        inp, mean, conv, bn = o[below(n)]["h"], below_mean(n), bl.conv1, bl.bn1
        dev = inp - mean.view(1, -1, 1, 1)
        w = conv.weight[n["c"]]
        c = inp.shape[1]
        g = grid(n["s"])
        r, q = torch.as_tensor(pos // g), torch.as_tensor(pos % g)
        at = torch.arange(len(idx))
        out = []
        for dy in range(3):
            for dx in range(3):
                mask = torch.zeros_like(w)
                mask[:, dy, dx] = 1
                y = F.conv2d(dev, (w * mask).view(c, 1, 3, 3), stride=conv.stride, padding=conv.padding, groups=c) * scale(bn)[n["c"]]
                out.append(y[at, :, r, q])
    return torch.cat(out, 1).numpy().astype(np.float64)


def score(Dr, Ds, z):
    order = np.argsort(-np.abs(Dr).mean(0))
    w = np.abs(Ds).mean(0)
    cum = np.cumsum(w[order])
    res = dict(n50=int(np.searchsorted(cum, .5 * cum[-1]) + 1), inputs=Ds.shape[1])
    sig, z2 = float((Ds.sum(1) ** 2).mean()), float((z ** 2).mean())
    for k in KS:
        e = float((Ds[:, order[k:]].sum(1) ** 2).mean())
        res[f"signal{k}"], res[f"share{k}"] = e / (sig + 1e-12), e / (z2 + 1e-12)
    return res


rng = np.random.default_rng(2)
out, count = {}, 2 if smoke else 15
for s in (4, 3, 2, 1):
    chans = rng.choice(WIDTH[s], count, replace=False).tolist()
    for b in (1, 0):
        for inner in (False, True):
            tag = f"s{s}b{b}{'i' if inner else ''}"
            rows, win = [], []
            for c in chans:
                n = parse(f"{tag}:{c}")
                ri, rp = top("read", n)
                si, sp = top("sealed", n)
                Dr = exact_inputs(m, PH["read"].tensor(ri), n, rp)[0]
                Ds, _, _, z = exact_inputs(m, PH["sealed"].tensor(si), n, sp)
                rows.append(score(Dr, Ds, z))
                if inner and s >= 3:
                    win.append(score(by_window("read", n, ri, rp), by_window("sealed", n, si, sp), z))
            med = {k: float(np.median([x[k] for x in rows])) for k in rows[0]}
            out[tag] = dict(median=med, rows=rows, channels=chans)
            print(tag, {k: round(v, 3) for k, v in med.items()}, flush=True)
            if win:
                mw = {k: float(np.median([x[k] for x in win])) for k in win[0]}
                out[tag]["window"] = mw
                print("   by window", {k: round(v, 3) for k, v in mw.items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "results", "one_layer_map" + ("_smoke" if smoke else "") + ".json"), "w"), indent=0)
