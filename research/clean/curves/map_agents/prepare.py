# serves: H-map
"""Cache for the map agents (no grouping, no clustering): for each photo of P, each channel of the last hidden layer:
h (mean over 7 x 7 after ReLU, what the logits read), z (mean pre-activation, for the negative pole), m (largest value
over the positions), and the logits. P = 20 ImageNet validation crops per class (20000 photos), from the first of
cache/r18/ix_train. These are not N's training photos. Usage: prepare.py resnet18|resnet50 [--smoke]"""
import os
import sys

import numpy as np
import torch
import torchvision

HERE = os.path.dirname(os.path.abspath(__file__))
CURVES = os.path.dirname(HERE)
OUT = os.path.join(CURVES, "cache", "map")
os.makedirs(OUT, exist_ok=True)
net = sys.argv[1]
smoke = "--smoke" in sys.argv
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
X = np.load(os.path.join(CURVES, "..", "hf", "imagenet", "val_224.npy"), mmap_mode="r")
ix = np.load(os.path.join(CURVES, "cache/r18/ix_train.npy"))
lab = np.load(os.path.join(CURVES, "cache/r18/lab_train.npy"))
keep = []
for c in range(1000):
    keep.extend(np.where(lab == c)[0][:20].tolist())
keep = np.array(sorted(keep))
photos, labels = ix[keep], lab[keep]
if smoke:
    photos, labels = photos[:64], labels[:64]
m = getattr(torchvision.models, net)(weights="IMAGENET1K_V1").eval().to(DEV)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEV)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEV)
blocks = list(m.layer4)
last = blocks[-1]
cap = {}
last.relu.register_forward_pre_hook(lambda mod, inp: cap.__setitem__("z", inp[0].clone()))  # relu is in place
Hs, Zs, Ms, Ys = [], [], [], []
with torch.no_grad():
    for b in range(0, len(photos), 64):
        sel = photos[b:b + 64]
        order = np.argsort(sel)
        inv = np.argsort(order)
        xb = torch.from_numpy(np.ascontiguousarray(X[sel[order]])).permute(0, 3, 1, 2).float().to(DEV) / 255.0
        xb = (xb - MEAN) / STD
        h = m.maxpool(m.relu(m.bn1(m.conv1(xb))))
        h = m.layer4(m.layer3(m.layer2(m.layer1(h))))
        # the block calls relu more than once; the last call before the output is the sum with the skip
        z = cap["z"]
        Hs.append(h.mean((2, 3)).cpu().numpy()[inv]); Zs.append(z.mean((2, 3)).cpu().numpy()[inv])
        Ms.append(h.amax((2, 3)).cpu().numpy()[inv]); Ys.append(m.fc(h.mean((2, 3))).cpu().numpy()[inv])
        if (b // 64) % 50 == 0:
            print(net, b, flush=True)
H, Z, M, Y = [np.concatenate(a).astype(np.float32) for a in (Hs, Zs, Ms, Ys)]
tag = net + ("_smoke" if smoke else "")
np.savez(os.path.join(OUT, f"{tag}_meta.npz"), photos=photos, labels=labels, W=m.fc.weight.detach().cpu().numpy(),
         b=m.fc.bias.detach().cpu().numpy())
for name, a in dict(h=H, z=Z, m=M, y=Y).items():
    np.save(os.path.join(OUT, f"{tag}_{name}.npy"), a)
chk = np.abs(H @ m.fc.weight.detach().cpu().numpy().T + m.fc.bias.detach().cpu().numpy() - Y).max()
print("done", net, H.shape, "logit check", float(chk), "share of channels with negative mean z", float((Z < 0).mean()), flush=True)
