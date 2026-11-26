"""N11 shared parts: network B = ConvNeXt-Base with the 3 stage-4 blocks (27, 28, 29) rebuilt narrow.

A = ConvNeXt-Base (torchvision). B keeps every A weight except the two MLP Linears of each stage-4 block:
  u = LayerNorm(dwconv(stream)),  g = GELU(W1 u + b1),  y = layer_scale * (W2 g + b2),  stream += y.
In B, W1 (k x 1024), b1, W2 (1024 x k), b2 start from random (torch default init, fixed seed) and are trained
by KL to A's class scores. Everything before stage 4 is A's and frozen, so B's block 27 reads A's exact input.

Cache (cache4.py): x4 = the stream at the input of stage 4's first block (after the downsample layer),
fp16, 1024 x 7 x 7, for every photo set of N6 (gap/common.py photo_sets), with A's logits and photo index.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gap"))
import common as C  # noqa: E402

DEV = C.DEV
OUT = os.path.join(HERE, "..", "results", "coarse")
CACHE4 = os.path.join(OUT, "cache4")
os.makedirs(OUT, exist_ok=True)
SETS = ("train", "trainflip", "held", "cal", "test")


def stage4(m):
    return list(m.features[7])


def head(m, x):
    return m.classifier(m.avgpool(x))


def run4(m, x4):
    """Stage 4 and the head, from the stream at the input of stage 4."""
    return head(m, m.features[7](x4))


def make_b(base, k, seed, bypass=False):
    """A copy of A whose stage-4 MLPs have k hidden neurons, random init (seed)."""
    import copy
    m = copy.deepcopy(base)
    g = torch.Generator().manual_seed(seed)
    for blk in stage4(m):
        d = blk.block[3].in_features
        l1, l2 = nn.Linear(d, k), nn.Linear(k, d)
        for lin in (l1, l2):
            bound = 1 / np.sqrt(lin.in_features)
            with torch.no_grad():
                lin.weight.uniform_(-bound, bound, generator=g)
                lin.bias.uniform_(-bound, bound, generator=g)
        blk.block[3], blk.block[5] = l1.to(DEV), l2.to(DEV)
    if bypass:
        raise NotImplementedError("bypass-blocked variant: only if the gate fails")
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def mlp_params(m):
    ps = []
    for blk in stage4(m):
        ps += [blk.block[3].weight, blk.block[3].bias, blk.block[5].weight, blk.block[5].bias]
    return ps


def mlp_state(m):
    return [{k: v.detach().cpu().clone() for k, v in (("W1", b.block[3].weight), ("b1", b.block[3].bias),
                                                      ("W2", b.block[5].weight), ("b2", b.block[5].bias))}
            for b in stage4(m)]


def load_mlp_state(m, st):
    with torch.no_grad():
        for b, d in zip(stage4(m), st):
            b.block[3].weight.copy_(d["W1"])
            b.block[3].bias.copy_(d["b1"])
            b.block[5].weight.copy_(d["W2"])
            b.block[5].bias.copy_(d["b2"])


class Cache4:
    """Stage-4 inputs (fp16), A's logits and photo index of one photo set. gpu=True loads it all to the device."""

    def __init__(self, name, gpu=False):
        x = np.load(os.path.join(CACHE4, f"{name}_x4.npy"), mmap_mode="r")
        self.z = torch.from_numpy(np.load(os.path.join(CACHE4, f"{name}_logits.npy"))).float()
        self.ix = np.load(os.path.join(CACHE4, f"{name}_ix.npy"))
        self.gpu = gpu
        if gpu:
            self.x = torch.from_numpy(np.ascontiguousarray(x)).to(DEV)
            self.z = self.z.to(DEV)
        else:
            self.x = x

    def __len__(self):
        return len(self.ix)

    def get(self, rows):
        rows = np.sort(np.asarray(rows))
        if self.gpu:
            r = torch.as_tensor(rows, device=DEV)
            return self.x[r].float(), self.z[r]
        x = torch.from_numpy(np.ascontiguousarray(self.x[rows])).to(DEV).float()
        return x, self.z[rows].to(DEV)


def evaluate(m, cache, labels=None, bs=200, rows_all=None):
    rows_all = np.arange(len(cache)) if rows_all is None else np.sort(rows_all)
    kl, hit = [], []
    with torch.no_grad():
        for a in range(0, len(rows_all), bs):
            rows = rows_all[a:a + bs]
            x, z = cache.get(rows)
            y = run4(m, x)
            kl.append(C.kl_rows(z, y).cpu())
            if labels is not None:
                hit.append((y.argmax(1).cpu() == torch.as_tensor(labels[cache.ix[rows]])).float())
    kl = torch.cat(kl)
    return dict(kl=float(kl.mean()), top1=float(torch.cat(hit).mean()) if hit else None), kl.numpy()
