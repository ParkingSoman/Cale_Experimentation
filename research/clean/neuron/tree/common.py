"""Shared parts for N5 (shrink one neuron's tree, n layers deep). ConvNeXt-Base, stages 3 and 4.

Global block index g: 0..26 = stage 3 blocks 0..26 (hidden 2,048, 14 x 14), 27..29 = stage 4 blocks 0..2 (hidden 4,096, 7 x 7).
A hidden neuron = one output of a block's first Linear; its output after GELU is hooked at blk.block[4].
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pair"))
from cache import DEV, load, net, split  # noqa: E402,F401

OUT = os.path.join(HERE, "..", "results", "tree")
os.makedirs(OUT, exist_ok=True)

VEHICLES = ['ambulance', 'beach wagon', 'bicycle-built-for-two', 'bullet train', 'cab', 'convertible',
            'electric locomotive', 'fire engine', 'forklift', 'freight car', 'garbage truck', 'go-kart', 'golfcart',
            'half track', 'jeep', 'jinrikisha', 'limousine', 'minibus', 'minivan', 'Model T', 'moped', 'motor scooter',
            'mountain bike', 'moving van', 'passenger car', 'pickup', 'police van', 'racer', 'recreational vehicle',
            'school bus', 'snowplow', 'sports car', 'steam locomotive', 'streetcar', 'tank', 'tow truck', 'tractor',
            'trailer truck', 'tricycle', 'trolleybus', 'unicycle']


def groups():
    names = net.class_names()
    return {
        "dogs": list(range(151, 269)),
        "birds": list(range(7, 25)) + list(range(80, 101)) + list(range(127, 147)),
        "vehicles": [i for i, c in enumerate(names) if c in VEHICLES],
    }


def blocks(m):
    return [blk for s in (5, 7) for blk in m.features[s]]


class Fills:
    """Replace chosen hidden neurons' GELU outputs with constants, in any blocks. fills[g] = (idx LongTensor, vals)."""

    def __init__(self, m):
        self.fills = {}
        self.hs = [blk.block[4].register_forward_hook(lambda mod, i, o, g=g: self._f(g, o))
                   for g, blk in enumerate(blocks(m))]

    def _f(self, g, out):
        if g not in self.fills:
            return out
        idx, vals = self.fills[g]
        out = out.clone()
        out[..., idx] = vals
        return out

    def set(self, d):
        self.fills = {g: (torch.as_tensor(i, device=DEV).long(), torch.as_tensor(v, device=DEV).float())
                      for g, (i, v) in d.items()}

    def clear(self):
        self.fills = {}


class Peaks:
    """Record each block's per-photo peak (max over positions) of the GELU output, for chosen blocks."""

    def __init__(self, m, which=None):
        self.store = {}
        self.which = set(range(30)) if which is None else set(which)
        self.hs = [blk.block[4].register_forward_hook(lambda mod, i, o, g=g: self._f(g, o))
                   for g, blk in enumerate(blocks(m))]

    def _f(self, g, out):
        if g in self.which:
            self.store[g] = out.amax((1, 2))
        return None


def kl_rows(ref_logits, logits):
    p = torch.log_softmax(ref_logits, 1)
    q = torch.log_softmax(logits, 1)
    return (p.exp() * (p - q)).sum(1)


def coarse_kl_rows(ref_logits, logits, grp):
    """KL of the 2-way split: probability inside the group against outside."""
    gi = torch.as_tensor(grp, device=ref_logits.device)

    def two(l):
        lp = torch.log_softmax(l, 1)
        a = torch.logsumexp(lp[:, gi], 1)
        b = torch.log1p(-a.exp().clamp(max=1 - 1e-7))
        return torch.stack([a, b], 1)
    p, q = two(ref_logits), two(logits)
    return (p.exp() * (p - q)).sum(1)
