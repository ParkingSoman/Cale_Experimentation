# serves: H-map
"""A reference, not a pass line: the one-layer measure on ConvNeXt-Tiny's last block, a network the essay calls clean.
Block: x + gamma * fc2(gelu(fc1(ln(dwconv7(x))))).   Neurons are the 3072 units of fc1, before their GELU.
   A stream direction d at the output: x_d + gamma_d * sum over neurons of fc2[d, j] * gelu_j. Inputs: 3072 neurons and the skip as one input.
   A neuron j: fc1[j] . ln(...) + bias. Inputs: 768 directions.
Reference means come from the list photos. 6 photos per class, split 3 and 3 inside the class. Lists on the first half, numbers on the second.
Its photos are ImageNet validation crops. Runs in torch on the CPU, from the same weights. Usage: reference_convnext.py [--smoke]"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


def load_file(path):
    """The safetensors format: an 8-byte header length, a JSON header, then the raw arrays. This venv does not have the package."""
    with open(path, "rb") as f:
        size = int.from_bytes(f.read(8), "little")
        head = json.loads(f.read(size))
        raw = f.read()
    out = {}
    for k, v in head.items():
        if k == "__metadata__":
            continue
        assert v["dtype"] == "F32", (k, v["dtype"])
        a, b = v["data_offsets"]
        out[k] = np.frombuffer(raw[a:b], dtype=np.float32).reshape(v["shape"])
    return out


HERE = os.path.dirname(os.path.abspath(__file__))
HF = os.path.join(HERE, "..", "..", "..", "hf")
smoke = "--smoke" in sys.argv
torch.set_num_threads(4)
w = {k: torch.as_tensor(v.copy()) for k, v in load_file(os.path.join(HF, "convnext", "tiny", "model.safetensors")).items()}
DEPTHS = (3, 3, 9, 3)
LAST = DEPTHS[3] - 1


def ln(x, wt, bi):
    mu, var = x.mean(-1, keepdim=True), x.var(-1, unbiased=False, keepdim=True)
    return (x - mu) / torch.sqrt(var + 1e-6) * wt + bi


def block_parts(x, s, b):
    """x is NCHW. Returns pre (NHWC), the neurons before GELU, their values after GELU, and the block's output."""
    p = f"stages.{s}.blocks.{b}."
    d = F.conv2d(x, w[p + "conv_dw.weight"], w[p + "conv_dw.bias"], padding=3, groups=x.shape[1]).permute(0, 2, 3, 1)
    pre = ln(d, w[p + "norm.weight"], w[p + "norm.bias"])
    a = pre @ w[p + "mlp.fc1.weight"].T + w[p + "mlp.fc1.bias"]
    g = F.gelu(a)
    out = x + ((g @ w[p + "mlp.fc2.weight"].T + w[p + "mlp.fc2.bias"]) * w[p + "gamma"]).permute(0, 3, 1, 2)
    return pre, a, g, out


def upto_last(x):
    x = F.conv2d(x, w["stem.0.weight"], w["stem.0.bias"], stride=4).permute(0, 2, 3, 1)
    x = ln(x, w["stem.1.weight"], w["stem.1.bias"]).permute(0, 3, 1, 2)
    for s in range(4):
        if s:
            y = ln(x.permute(0, 2, 3, 1), w[f"stages.{s}.downsample.0.weight"], w[f"stages.{s}.downsample.0.bias"]).permute(0, 3, 1, 2)
            x = F.conv2d(y, w[f"stages.{s}.downsample.1.weight"], w[f"stages.{s}.downsample.1.bias"], stride=2)
        for b in range(DEPTHS[s]):
            if s == 3 and b == LAST:
                return x
            x = block_parts(x, s, b)[3]


X = np.load(os.path.join(HF, "imagenet", "val_224.npy"), mmap_mode="r")
lab = np.load(os.path.join(HF, "imagenet", "val_labels.npy"))
per = 1 if smoke else 3
A = [i for c in range(1000) for i in np.where(lab == c)[0][:per]]
B = [i for c in range(1000) for i in np.where(lab == c)[0][per:2 * per]]
if smoke:
    A, B = A[:128], B[:128]
MEAN, STD = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1), torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)


def run(ids):
    xs, ps, ass, gs, os_ = [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(ids), 32):
            sel = [int(q) for q in ids[i:i + 32]]
            x = (torch.from_numpy(np.stack([np.asarray(X[q]) for q in sel])).permute(0, 3, 1, 2).float() / 255 - MEAN) / STD
            h = upto_last(x)
            pre, a, g, o = block_parts(h, 3, LAST)
            xs.append(h.half())
            ps.append(pre.half())
            ass.append(a.half())
            gs.append(g.half())
            os_.append(o.half())
    return [torch.cat(v).float() for v in (xs, ps, ass, gs, os_)]


hA, preA, aA, gA, oA = run(A)
hB, preB, aB, gB, oB = run(B)
gm, pm, hm = gA.mean((0, 1, 2)), preA.mean((0, 1, 2)), hA.mean((0, 2, 3))
P = f"stages.3.blocks.{LAST}."
fc1, fc2, gamma = w[P + "mlp.fc1.weight"], w[P + "mlp.fc2.weight"], w[P + "gamma"]
KS = (4, 16, 64)


def peak(field):
    """field N x H x W. Top 64 photos by largest value, with their positions."""
    v, p = field.flatten(1).max(1)
    idx = torch.argsort(v, descending=True)[:64]
    return idx, torch.div(p[idx], field.shape[2], rounding_mode="floor"), p[idx] % field.shape[2]


def direction(d, h, g, o):
    idx, r, q = peak(o[:, d])
    D = (g[idx, r, q] - gm) * fc2[d] * gamma[d]
    Ds = (h[idx, d, r, q] - hm[d])[:, None]
    return torch.cat([D, Ds], 1).numpy().astype(np.float64), o[idx, d, r, q].numpy()


def neuron(j, pre, a):
    idx, r, q = peak(a[:, :, :, j])
    return ((pre[idx, r, q] - pm) * fc1[j]).numpy().astype(np.float64), a[idx, r, q, j].numpy()


def score(Dr, Ds, z):
    order = np.argsort(-np.abs(Dr).mean(0))
    wt = np.abs(Ds).mean(0)
    cum = np.cumsum(wt[order])
    res = dict(n50=int(np.searchsorted(cum, .5 * cum[-1]) + 1), inputs=Ds.shape[1])
    sig, z2 = float((Ds.sum(1) ** 2).mean()), float((z ** 2).mean())
    for k in KS:
        e = float((Ds[:, order[k:]].sum(1) ** 2).mean())
        res[f"signal{k}"], res[f"share{k}"] = e / (sig + 1e-12), e / (z2 + 1e-12)
    return res


rng = np.random.default_rng(3)
count = 2 if smoke else 30
rows_d = [score(direction(int(d), hA, gA, oA)[0], *direction(int(d), hB, gB, oB)) for d in rng.choice(768, count, replace=False)]
rows_n = [score(neuron(int(j), preA, aA)[0], *neuron(int(j), preB, aB)) for j in rng.choice(3072, count, replace=False)]
res = {}
for name, rows in (("stream direction from its neurons", rows_d), ("neuron from its stream directions", rows_n)):
    res[name] = {k: float(np.median([x[k] for x in rows])) for k in rows[0]}
    print(name, {k: round(v, 3) for k, v in res[name].items()}, flush=True)
json.dump(res, open(os.path.join(HERE, "results", "reference_convnext" + ("_smoke" if smoke else "") + ".json"), "w"), indent=0)
