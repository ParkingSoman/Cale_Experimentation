# serves: H-map
"""Cache for the ResNet-18 zoom (plan okay-i-really-enjoyed-humble-allen.md, step 3).

For every image of the train split (35 per class) and the sealed split (10 per class) of `inception.split`:
  z   the pre-activation of ResNet-18's last block, z = bn2(conv2(.)) + skip, before the final ReLU (512 x 7 x 7), float16
  y   N's logits (1000), float32
Files in cache/r18/: z_<split>.npy (memmap), y_<split>.npy, ix_<split>.npy (image index into val_224), lab_<split>.npy.
Usage: python rz_cache.py [n_images_per_split_for_smoke]
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import resnets as R

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "r18")
os.makedirs(OUT, exist_ok=True)
smoke = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def zlast(m, x):
    h = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    h = m.layer4[0](m.layer3(m.layer2(m.layer1(h))))
    b = m.layer4[1]
    return b.bn2(b.conv2(b.relu(b.bn1(b.conv1(h))))) + h


m = R.load("resnet18")
X, y = R.load_images()
tr, _, se = R.split(y)
for name, ix in (("train", tr), ("sealed", se)):
    if smoke:
        ix = np.random.default_rng(0).choice(ix, smoke, replace=False)
    ix = np.sort(ix)
    tag = name + ("_smoke" if smoke else "")
    Z = np.lib.format.open_memmap(os.path.join(OUT, f"z_{tag}.npy"), mode="w+", dtype=np.float16, shape=(len(ix), 512, 49))
    Y = np.zeros((len(ix), 1000), np.float32)
    t0 = time.time(); bs = 100; maxdiff = 0.0
    with torch.no_grad():
        for b in range(0, len(ix), bs):
            x = R.prep(X[ix[b:b + bs]])
            z = zlast(m, x)
            yy = R.head(m, torch.relu(z))
            if b == 0:
                ref = R.head(m, R.last_layer(m, x))
                maxdiff = float((ref - yy).abs().max())
            Z[b:b + len(x)] = z.reshape(len(x), 512, 49).cpu().numpy().astype(np.float16)
            Y[b:b + len(x)] = yy.cpu().numpy()
            if (b // bs) % 50 == 0:
                print(name, b, len(ix), f"{time.time()-t0:.0f}s", flush=True)
    Z.flush(); del Z
    np.save(os.path.join(OUT, f"y_{tag}.npy"), Y)
    np.save(os.path.join(OUT, f"ix_{tag}.npy"), ix)
    np.save(os.path.join(OUT, f"lab_{tag}.npy"), y[ix])
    top1 = float((Y.argmax(1) == y[ix]).mean())
    print(name, "n", len(ix), "top1", round(top1, 4), "max |logit diff| vs torchvision path", maxdiff, f"{time.time()-t0:.0f}s", flush=True)
