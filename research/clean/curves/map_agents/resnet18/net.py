# serves: H-map
"""ResNet-18 written out by hand, so every object a map talks about has a name. No grouping and no ranking of groups here.

Names, for torchvision's resnet18 IMAGENET1K_V1:
   stem      conv1, bn1, relu, maxpool: 64 channels. It feeds stage 1's skip with no change.
   stage s   s = 1, 2, 3, 4, with 64, 128, 256, 512 channels and blocks b = 0, 1.
   inner     relu(bn1(conv1(input))): the channels between the two convolutions of a block.
   r         bn2(conv2(inner)): the block's residual path.
   skip      the input, or downsample(input) for block 0 of stages 2, 3, 4.
   z         r + skip, before the ReLU.
   h         relu(z). The stream.
A stream channel "s4:179" means channel 179 of stage 4. Its last value in the stage is h at block 1.
The input and the stream of a stage: h of stage s, block 1, is the input of stage s + 1, block 0."""
import os

import numpy as np
import torch
import torchvision

HERE = os.path.dirname(os.path.abspath(__file__))
CURVES = os.path.dirname(os.path.dirname(HERE))
STORE = os.path.join(CURVES, "cache", "map18")
WIDTH = {0: 64, 1: 64, 2: 128, 3: 256, 4: 512}
STRIDE = {0: 4, 1: 4, 2: 8, 3: 16, 4: 32}
CROP = {0: 40, 1: 56, 2: 88, 3: 112, 4: 140}  # crop size around a peak, within the region a channel reads


def load():
    m = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def stages(m):
    return {1: m.layer1, 2: m.layer2, 3: m.layer3, 4: m.layer4}


def stem(m, x):
    return m.maxpool(torch.relu(m.bn1(m.conv1(x))))


def block_parts(bl, h):
    """One BasicBlock by hand. Same arithmetic as torchvision's forward."""
    inner = torch.relu(bl.bn1(bl.conv1(h)))
    r = bl.bn2(bl.conv2(inner))
    skip = h if bl.downsample is None else bl.downsample(h)
    z = r + skip
    return dict(inner=inner, r=r, skip=skip, z=z, h=torch.relu(z))


def head(m, h):
    return m.fc(torch.flatten(m.avgpool(h), 1))


def forward(m, x, fill=None):
    """Whole network. Returns logits and every block's parts.
    fill: optional {stage: (channels, values at block 0, values at block 1)}. Those stream channels become those constants at every
    position, at both blocks, so a removal goes through the whole stage. Each block has its own mean (Definition DC1). The stem is stage 0,
    block 0: its values at block 1 are not used."""
    def apply(h, s, b):
        if fill and s in fill:
            ch, val = fill[s][0], fill[s][1 + b]
            h = h.clone()
            h[:, ch] = val.to(h).view(1, -1, 1, 1)
        return h
    out = {}
    h = apply(stem(m, x), 0, 0)
    out[(0, 0)] = dict(h=h)
    for s, lay in stages(m).items():
        for b, bl in enumerate(lay):
            p = block_parts(bl, h)
            p["h"] = apply(p["h"], s, b)
            h = p["h"]
            out[(s, b)] = p
    return head(m, h), out


def upto(m, x, s):
    """The input of stage s: h of stage s - 1, block 1 (the stem for s = 1)."""
    h = stem(m, x)
    for t, lay in stages(m).items():
        if t == s:
            return h
        h = lay(h)
    raise ValueError(s)


def run_stage(m, s, h):
    """Stage s from its input. Returns the parts of both blocks."""
    lay = stages(m)[s]
    p0 = block_parts(lay[0], h)
    p1 = block_parts(lay[1], p0["h"])
    return p0, p1


class Photos:
    """The reading photos and the sealed photos. They never mix. Lists are sorted by class, so split inside a class."""

    def __init__(self, which="read"):
        assert which in ("read", "sealed")
        self.which = which
        name = "train" if which == "read" else "sealed"
        self.ix = np.load(os.path.join(CURVES, f"cache/r18/ix_{name}.npy"))
        self.lab = np.load(os.path.join(CURVES, f"cache/r18/lab_{name}.npy"))
        self.X = np.load(os.path.join(CURVES, "..", "hf", "imagenet", "val_224.npy"), mmap_mode="r")
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.ix)

    def raw(self, idx):
        return np.stack([np.asarray(self.X[int(self.ix[i])]) for i in idx])

    def tensor(self, idx, raw=None):
        raw = self.raw(idx) if raw is None else raw
        x = torch.from_numpy(raw.copy()).permute(0, 3, 1, 2).float() / 255.0
        return (x - self.mean) / self.std

    def per_class(self, n):
        out = []
        for c in range(1000):
            out.extend(np.where(self.lab == c)[0][:n].tolist())
        return np.array(sorted(out))


def class_names():
    return torchvision.models.ResNet18_Weights.IMAGENET1K_V1.meta["categories"]


def block_means(read_dir):
    """Mean of each stream channel, per stage and block, over the reading photos: {stage: (block 0 means, block 1 means)}."""
    out = {0: None}
    a = np.asarray(np.load(os.path.join(read_dir, "s0b0_hm.npy"), mmap_mode="r"), dtype=np.float32).mean(0)
    out[0] = (torch.as_tensor(a), torch.as_tensor(a))
    for s in (1, 2, 3, 4):
        out[s] = tuple(torch.as_tensor(np.asarray(np.load(os.path.join(read_dir, f"s{s}b{b}_hm.npy"), mmap_mode="r"), dtype=np.float32).mean(0)) for b in (0, 1))
    return out


def parse(ch):
    """'s4:179' -> (4, 179)."""
    s, c = ch.split(":")
    return int(s.lstrip("s")), int(c)
