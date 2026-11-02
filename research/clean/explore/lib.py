"""Lab helpers for exploring InceptionV1 by hand (Phase 1 of the exploration plan). Not part of the paper.

Reuses research/clean/curves/inception.py. Adds: every block output, any block's pre-ReLU, class names, and
contact sheets of images (so the explorer can look at them).
"""
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
CURVES = os.path.join(HERE, "..", "curves")
sys.path.insert(0, CURVES)
from inception import DEV, load_images, load_model, mixed, prep, to_3a  # noqa: E402,F401

OUT = os.path.join(HERE, "out")
BLOCKS = ["3a", "3b", "4a", "4b", "4c", "4d", "4e", "5a", "5b"]
# branch widths (1x1, 3x3, 5x5, pool) of each block, from the lucid model
_POOL_BEFORE = {"4a": "maxpool4", "5a": "maxpool10"}


def mixed_pre(m, name, h):
    """Pre-ReLU of every channel of one inception block, concatenated like the block output."""
    g = lambda s: getattr(m, f"{name}_{s}")
    a = g("1x1_pre_relu_conv")(h)
    b = g("3x3_bottleneck")(g("3x3_bottleneck_pre_relu_conv")(h))
    c = g("5x5_bottleneck")(g("5x5_bottleneck_pre_relu_conv")(h))
    p = g("pool")(F.pad(h, (1, 1, 1, 1), value=float('-inf')), kernel_size=(3, 3), stride=(1, 1), padding=0,
                  ceil_mode=False)
    d = g("pool_reduce_pre_relu_conv")(p)
    b = g("3x3_pre_relu_conv")(F.pad(b, (1, 1, 1, 1)))
    c = g("5x5_pre_relu_conv")(F.pad(c, (2, 2, 2, 2)))
    return torch.cat([a, b, c, d], 1)


def _pool(m, which, h):
    return getattr(m, which)(F.pad(h, (0, 1, 0, 1), value=float('-inf')), kernel_size=(3, 3), stride=(2, 2),
                             padding=0, ceil_mode=False)


def run(m, x, want_pre=(), edit=None):
    """Return dict: block name -> output (post-ReLU), 'in_<block>' -> the block's input, 'pre_<block>' for blocks in
    want_pre, and 'logits'. edit: {block: fn} replaces a block's output by fn(output) before the next block reads it."""
    edit = edit or {}
    out = {}
    h = to_3a(m, x)
    if "3a" in edit:
        h = edit["3a"](h)
    out["3a"] = h
    for name in BLOCKS[1:]:
        if name in _POOL_BEFORE:
            h = _pool(m, _POOL_BEFORE[name], h)
        out["in_" + name] = h
        if name in want_pre:
            out["pre_" + name] = mixed_pre(m, "mixed" + name, h)
        h = mixed(m, "mixed" + name, h)
        if name in edit:
            h = edit[name](h)
        out[name] = h
    hp = F.avg_pool2d(h, kernel_size=(7, 7), stride=(1, 1))
    out["logits"] = m.softmax2_pre_activation_matmul(hp.reshape(-1, 1024))
    return out


def class_names():
    """ImageNet class names in the dataset's label order (from the cached dataset README)."""
    names = {}
    for line in open(os.path.join(HERE, "..", "hf", "imagenet", "README.md")):
        mm = re.match(r"\s+'(\d+)': (.*)$", line)
        if mm:
            names[int(mm.group(1))] = mm.group(2).strip()
    return [names[i] for i in range(len(names))]


def sheet(images, captions, path, cols=6, size=160, boxes=None):
    """Save a contact sheet: images (uint8 HWC), one caption each, optional boxes (x0, y0, x1, y1) in 224 pixels."""
    rows = (len(images) + cols - 1) // cols
    W = Image.new("RGB", (cols * size, rows * (size + 28)), "white")
    d = ImageDraw.Draw(W)
    for k, (im, cap) in enumerate(zip(images, captions)):
        r, c = divmod(k, cols)
        pim = Image.fromarray(np.asarray(im)).resize((size, size))
        if boxes is not None and boxes[k] is not None:
            dd = ImageDraw.Draw(pim)
            s = size / 224.0
            x0, y0, x1, y1 = boxes[k]
            dd.rectangle([x0 * s, y0 * s, x1 * s, y1 * s], outline=(255, 0, 0), width=2)
        W.paste(pim, (c * size, r * (size + 28)))
        d.text((c * size + 2, r * (size + 28) + size + 2), cap[:28], fill="black")
        d.text((c * size + 2, r * (size + 28) + size + 14), cap[28:56], fill="black")
    W.save(path)
    return path
