"""Lucid InceptionV1 (lucent port), split at layer 3b, plus ImageNet data and the curves-track split.

Layer 3b = concat of four branches (channel ranges in the 480 output):
  1x1 0-127 | 3x3 128-319 | 5x5 320-415 | pool 416-479
The ten curve detectors (Distill "Curve Detectors") all sit in the 5x5 branch.
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "vision2"))
HF = os.path.join(HERE, "..", "hf")
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

CURVES = [379, 406, 385, 343, 342, 388, 324, 340, 330, 349]
BRANCH = {"1x1": (0, 128), "3x3": (128, 320), "5x5": (320, 416), "pool": (416, 480)}


def load_model(redirected=False):
    """redirected=True keeps lucent's feature-vis ReLU, whose backward leaks 0.1 x the gradient through switched-off
    units (a wrong gradient). Forward passes are identical. Found 2026-10-22: all training before then used it."""
    from lucent.modelzoo import inceptionv1
    m = inceptionv1(pretrained=True, redirected_ReLU=redirected).eval().to(DEV)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_images():
    """50k val, 224 center crop, uint8 NHWC (memory-mapped), labels."""
    xf, yf = os.path.join(HF, "imagenet", "val_224.npy"), os.path.join(HF, "imagenet", "val_labels.npy")
    return np.load(xf, mmap_mode="r"), np.load(yf)


def split(y):
    """Per class: 35 train, 5 dev, 10 sealed (by order inside the class)."""
    rank = np.zeros(len(y), int)
    for c in range(1000):
        ix = np.where(y == c)[0]; rank[ix] = np.arange(len(ix))
    return np.where(rank < 35)[0], np.where((rank >= 35) & (rank < 40))[0], np.where(rank >= 40)[0]


def prep(xb):
    """uint8 NHWC numpy -> float NCHW on device, lucid scale (0-255 minus 117)."""
    x = torch.from_numpy(np.ascontiguousarray(xb)).to(DEV).permute(0, 3, 1, 2).float()
    return x - 117.0


def to_3a(m, x):
    """Stem up to the 3a output (256 x 28 x 28)."""
    h = m.conv2d0(m.conv2d0_pre_relu_conv(F.pad(x, (2, 3, 2, 3))))
    h = m.maxpool0(F.pad(h, (0, 1, 0, 1), value=float('-inf')), kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
    h = F.local_response_norm(h, size=9, alpha=9.99999974738e-05, beta=0.5, k=1)
    h = m.conv2d1(m.conv2d1_pre_relu_conv(h))
    h = m.conv2d2(m.conv2d2_pre_relu_conv(F.pad(h, (1, 1, 1, 1))))
    h = F.local_response_norm(h, size=9, alpha=9.99999974738e-05, beta=0.5, k=1)
    h = m.maxpool1(F.pad(h, (0, 1, 0, 1), value=float('-inf')), kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
    return mixed(m, "mixed3a", h)


def mixed(m, name, h):
    """One inception block (ReLU outputs, concat)."""
    g = lambda s: getattr(m, f"{name}_{s}")
    a = g("1x1")(g("1x1_pre_relu_conv")(h))
    b = g("3x3_bottleneck")(g("3x3_bottleneck_pre_relu_conv")(h))
    c = g("5x5_bottleneck")(g("5x5_bottleneck_pre_relu_conv")(h))
    p = g("pool")(F.pad(h, (1, 1, 1, 1), value=float('-inf')), kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
    d = g("pool_reduce")(g("pool_reduce_pre_relu_conv")(p))
    b = g("3x3")(g("3x3_pre_relu_conv")(F.pad(b, (1, 1, 1, 1))))
    c = g("5x5")(g("5x5_pre_relu_conv")(F.pad(c, (2, 2, 2, 2))))
    return torch.cat([a, b, c, d], 1)


def inputs_3b(m, a3):
    """The four inputs the 3b branches read: 3a itself (1x1), 3x3 bottleneck (128), 5x5 bottleneck (32), pooled 3a (256)."""
    b = m.mixed3b_3x3_bottleneck(m.mixed3b_3x3_bottleneck_pre_relu_conv(a3))
    c = m.mixed3b_5x5_bottleneck(m.mixed3b_5x5_bottleneck_pre_relu_conv(a3))
    p = m.mixed3b_pool(F.pad(a3, (1, 1, 1, 1), value=float('-inf')), kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
    return a3, b, c, p


def pre_3b(m, a3):
    """3b pre-ReLU (480 x 28 x 28)."""
    a, b, c, p = inputs_3b(m, a3)
    return torch.cat([m.mixed3b_1x1_pre_relu_conv(a),
                      m.mixed3b_3x3_pre_relu_conv(F.pad(b, (1, 1, 1, 1))),
                      m.mixed3b_5x5_pre_relu_conv(F.pad(c, (2, 2, 2, 2))),
                      m.mixed3b_pool_reduce_pre_relu_conv(p)], 1)


def from_3b(m, h3b, upto="logits"):
    """3b output (480 x 28 x 28, post-ReLU) -> logits (1008, pre-softmax). upto='4a' returns the 4a output."""
    h = m.maxpool4(F.pad(h3b, (0, 1, 0, 1), value=float('-inf')), kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
    h = mixed(m, "mixed4a", h)
    if upto == "4a":
        return h
    for n in ["mixed4b", "mixed4c", "mixed4d", "mixed4e"]:
        h = mixed(m, n, h)
    h = m.maxpool10(F.pad(h, (0, 1, 0, 1), value=float('-inf')), kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
    for n in ["mixed5a", "mixed5b"]:
        h = mixed(m, n, h)
    h = F.avg_pool2d(h, kernel_size=(7, 7), stride=(1, 1))
    return m.softmax2_pre_activation_matmul(h.reshape(-1, 1024))
