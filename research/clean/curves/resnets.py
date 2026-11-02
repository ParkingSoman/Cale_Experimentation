"""torchvision ResNets (IMAGENET1K_V1 weights, one training recipe at five sizes) for Part C of COARSE-THEORY.md.

The last hidden layer is layer4's output (after its final ReLU, C x 7 x 7). The head is the mean over the 7 x 7
positions followed by one linear layer, so the logits are linear in every unit of that layer (the premise of Part B.M
and B.E). Images: the ImageNet val crops of `inception.load_images` (224 x 224, uint8, RGB).
"""
import numpy as np
import torch
import torchvision

from inception import DEV, load_images, split

SIZES = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load(name):
    m = getattr(torchvision.models, name)(weights="IMAGENET1K_V1").eval().to(DEV)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def prep(xb):
    x = torch.from_numpy(np.ascontiguousarray(xb)).permute(0, 3, 1, 2).float() / 255.0
    return ((x - MEAN) / STD).to(DEV)


def last_layer(m, x):
    """layer4 output (N, C, 7, 7), after the final ReLU."""
    h = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    return m.layer4(m.layer3(m.layer2(m.layer1(h))))


def head(m, H):
    """logits from the last hidden layer (N, C, 7, 7) or its pooled form (N, C)."""
    if H.dim() == 4:
        H = H.mean(dim=(2, 3))
    return m.fc(H)


def acts(m, X, ix, bs=50):
    """Last-layer activations (float32 numpy, N x C x 7 x 7) and logits for the images ix."""
    Hs, Ys = [], []
    with torch.no_grad():
        for b in range(0, len(ix), bs):
            sel = np.asarray(ix[b:b + bs])
            order = np.argsort(sel)
            inv = np.argsort(order)
            H = last_layer(m, prep(X[sel[order]]))
            Hs.append(H.cpu().numpy()[inv]); Ys.append(head(m, H).cpu().numpy()[inv])
    return np.concatenate(Hs), np.concatenate(Ys)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


__all__ = ["SIZES", "load", "prep", "last_layer", "head", "acts", "n_params", "load_images", "split", "DEV"]
