"""ConvNeXt-Tiny's last block, for the positive control. A network the essay calls clean. Block: x + gamma * fc2(gelu(fc1(ln(dwconv7(x))))). Neurons are the 3072 units of fc1, before their GELU.

The store: one array per split, one row per photo, with each neuron's strongest value and place on both poles, and its mean. Photos are the same reading and sealed lists that the ResNet-18 run uses, with a share of the reading list to keep the store small.
Run this file to build the store: net.py build"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.abspath(os.path.join(HERE, "..", "..", "..", "research", "clean"))
HF = os.path.join(CLEAN, "hf")
STORE = os.path.join(CLEAN, "curves", "cache", "mapper_convnext")
DEPTHS = (3, 3, 9, 3)
LAST = DEPTHS[3] - 1
GRID = 7
MEAN, STD = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1), torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
READ_PER_CLASS = 10
SEALED_PER_CLASS = 8


def load_weights():
    with open(os.path.join(HF, "convnext", "tiny", "model.safetensors"), "rb") as f:
        size = int.from_bytes(f.read(8), "little")
        head = json.loads(f.read(size))
        raw = f.read()
    out = {}
    for k, v in head.items():
        if k != "__metadata__":
            a, b = v["data_offsets"]
            out[k] = torch.as_tensor(np.frombuffer(raw[a:b], dtype=np.float32).reshape(v["shape"]).copy())
    return out


def ln(x, wt, bi):
    mu, var = x.mean(-1, keepdim=True), x.var(-1, unbiased=False, keepdim=True)
    return (x - mu) / torch.sqrt(var + 1e-6) * wt + bi


def block_parts(w, x, s, b):
    p = f"stages.{s}.blocks.{b}."
    d = F.conv2d(x, w[p + "conv_dw.weight"], w[p + "conv_dw.bias"], padding=3, groups=x.shape[1]).permute(0, 2, 3, 1)
    pre = ln(d, w[p + "norm.weight"], w[p + "norm.bias"])
    a = pre @ w[p + "mlp.fc1.weight"].T + w[p + "mlp.fc1.bias"]
    return pre, a


def after_block(w, x, a, s, b):
    p = f"stages.{s}.blocks.{b}."
    g = F.gelu(a)
    return x + ((g @ w[p + "mlp.fc2.weight"].T + w[p + "mlp.fc2.bias"]) * w[p + "gamma"]).permute(0, 3, 1, 2)


def upto_last(w, x):
    x = F.conv2d(x, w["stem.0.weight"], w["stem.0.bias"], stride=4).permute(0, 2, 3, 1)
    x = ln(x, w["stem.1.weight"], w["stem.1.bias"]).permute(0, 3, 1, 2)
    for s in range(4):
        if s:
            y = ln(x.permute(0, 2, 3, 1), w[f"stages.{s}.downsample.0.weight"], w[f"stages.{s}.downsample.0.bias"]).permute(0, 3, 1, 2)
            x = F.conv2d(y, w[f"stages.{s}.downsample.1.weight"], w[f"stages.{s}.downsample.1.bias"], stride=2)
        for b in range(DEPTHS[s]):
            if s == 3 and b == LAST:
                return x
            pre, a = block_parts(w, x, s, b)
            x = after_block(w, x, a, s, b)
    raise AssertionError


def head(w, x):
    pooled = x.mean((2, 3))
    y = ln(pooled, w["head.norm.weight"], w["head.norm.bias"]) if "head.norm.weight" in w else pooled
    return y @ w["head.fc.weight"].T + w["head.fc.bias"]


class Photos:
    def __init__(self, split):
        lists = os.path.join(CLEAN, "curves", "cache", "r18")
        name = "train" if split == "read" else "sealed"
        ix = np.load(os.path.join(lists, f"ix_{name}.npy"))
        lab = np.load(os.path.join(lists, f"lab_{name}.npy"))
        if split == "read":
            keep = np.concatenate([np.where(lab == c)[0][:READ_PER_CLASS] for c in range(1000)])
            ix, lab = ix[keep], lab[keep]
        else:
            keep = np.concatenate([np.where(lab == c)[0][:SEALED_PER_CLASS] for c in range(1000)])
            ix, lab = ix[keep], lab[keep]
        self.ix, self.lab = ix, lab
        self.X = np.load(os.path.join(HF, "imagenet", "val_224.npy"), mmap_mode="r")

    def raw(self, idx):
        return np.stack([np.asarray(self.X[int(self.ix[i])]) for i in idx])

    @staticmethod
    def tensor(raw):
        return (torch.from_numpy(raw.copy()).permute(0, 3, 1, 2).float() / 255 - MEAN) / STD


def build():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    w = {k: v.to(device) for k, v in load_weights().items()}
    os.makedirs(STORE, exist_ok=True)
    for split in ("read", "sealed"):
        P = Photos(split)
        n = len(P.ix)
        arrays = {k: np.lib.format.open_memmap(os.path.join(STORE, f"{split}_{k}.npy"), mode="w+", dtype=dt, shape=(n, 3072)) for k, dt in (("x", "f2"), ("p", "u1"), ("n", "f2"), ("q", "u1"), ("m", "f2"))}
        np.save(os.path.join(STORE, f"{split}_labels.npy"), P.lab)
        np.save(os.path.join(STORE, f"{split}_ix.npy"), P.ix)
        with torch.no_grad():
            for start in range(0, n, 64):
                sel = range(start, min(start + 64, n))
                x = Photos.tensor(P.raw(sel)).to(device)
                _, a = block_parts(w, upto_last(w, x), 3, LAST)
                flat = a.flatten(1, 2)
                vx, px = flat.max(1)
                vn, pn = flat.min(1)
                arrays["x"][start:start + len(sel)] = vx.cpu().numpy()
                arrays["p"][start:start + len(sel)] = px.cpu().numpy()
                arrays["n"][start:start + len(sel)] = vn.cpu().numpy()
                arrays["q"][start:start + len(sel)] = pn.cpu().numpy()
                arrays["m"][start:start + len(sel)] = flat.mean(1).cpu().numpy()
                if start % 3200 == 0:
                    print(split, start, n, flush=True)
        for v in arrays.values():
            v.flush()
    print("done", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
