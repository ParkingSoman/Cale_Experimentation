"""N11 shared parts on ResNet-18 (torchvision IMAGENET1K_V1) = network A.

Stage 4 (two BasicBlocks) from its input x (= stage-3 output h, 256 x 14 x 14):
  block 0: inner = relu(bn1(conv1 x)) [512 neurons, stride 2], r = bn2(conv2 inner), h = relu(r + downsample(x))
  block 1: inner = relu(bn1(conv1 h)) [512 neurons],           r = bn2(conv2 inner), h = relu(r + h)
  logits = fc(avgpool(h)).
Network B: the same, but each block's inner layer has k neurons (conv1, bn1, conv2, bn2 new, random init, seed).
The stream (512 channels), the downsample and the head stay A's, frozen.

Photos: the reading photos (35 per class, curves/map_agents/resnet18/net.py). Inside each class (sorted list):
even positions = fit, odd positions = test. held = fit positions 8 and 9 (2 per class). train = the other fit photos.
The sealed photos are never opened.
"""
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "curves", "map_agents", "resnet18"))
import net  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
OUT = os.path.join(HERE, "..", "results", "coarse", "r18")
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)
CARS = [407, 436, 468, 511, 609, 627, 656, 661, 717, 751, 817]
SETS = ("train", "trainflip", "held", "test")


def photo_sets(lab):
    tr, held, te = [], [], []
    for c in range(1000):
        w = np.where(lab == c)[0]
        fit, odd = w[0::2], w[1::2]
        held += fit[8:10].tolist()
        tr += [i for k, i in enumerate(fit) if k not in (8, 9)]
        te += odd.tolist()
    return dict(train=np.array(tr), held=np.array(held), test=np.array(te))


def load_a():
    m = net.load().to(DEV).eval()
    return m


def stage4(m):
    return m.layer4


def run4(m, x):
    return net.head(m, m.layer4(x))


def make_b(a, k, seed, blocks=(0, 1)):
    m = copy.deepcopy(a)
    torch.manual_seed(seed)
    for bi, bl in enumerate(m.layer4):
        if bi not in blocks:
            continue
        cin = bl.conv1.in_channels
        bl.conv1 = nn.Conv2d(cin, k, 3, stride=bl.conv1.stride, padding=1, bias=False)
        bl.bn1 = nn.BatchNorm2d(k)
        bl.conv2 = nn.Conv2d(k, 512, 3, padding=1, bias=False)
        bl.bn2 = nn.BatchNorm2d(512)
        for c in (bl.conv1, bl.conv2):
            nn.init.kaiming_normal_(c.weight, mode="fan_out", nonlinearity="relu")
    m = m.to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def new_modules(m, blocks=(0, 1)):
    return [mod for bi, bl in enumerate(m.layer4) if bi in blocks for mod in (bl.conv1, bl.bn1, bl.conv2, bl.bn2)]


def new_params(m, blocks=(0, 1)):
    return [p for mod in new_modules(m, blocks) for p in mod.parameters()]


def state(m):
    return {k: v.detach().cpu().clone() for k, v in m.layer4.state_dict().items()}


def load_state(m, st):
    m.layer4.load_state_dict(st)


class Cache:
    def __init__(self, name, gpu=False):
        x = np.load(os.path.join(CACHE, f"{name}_x.npy"), mmap_mode="r")
        self.z = torch.from_numpy(np.load(os.path.join(CACHE, f"{name}_logits.npy"))).float()
        self.ix = np.load(os.path.join(CACHE, f"{name}_ix.npy"))
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
        return torch.from_numpy(np.ascontiguousarray(self.x[rows])).to(DEV).float(), self.z[rows].to(DEV)


def kl_rows(z, y):
    p = F.log_softmax(z.float(), 1)
    return (p.exp() * (p - F.log_softmax(y.float(), 1))).sum(1)


def evaluate(m, cache, labels=None, bs=250, rows_all=None):
    rows_all = np.arange(len(cache)) if rows_all is None else np.sort(rows_all)
    kl, hit = [], []
    with torch.no_grad():
        for a in range(0, len(rows_all), bs):
            rows = rows_all[a:a + bs]
            x, z = cache.get(rows)
            y = run4(m, x)
            kl.append(kl_rows(z, y).cpu())
            if labels is not None:
                hit.append((y.argmax(1).cpu() == torch.as_tensor(labels[cache.ix[rows]])).float())
    kl = torch.cat(kl)
    return dict(kl=float(kl.mean()), top1=float(torch.cat(hit).mean()) if hit else None), kl.numpy()
