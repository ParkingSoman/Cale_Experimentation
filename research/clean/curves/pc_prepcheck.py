# serves: H-weak
"""Preprocessing check for pc_weak.py / pc_scale.py: ResNet-50 top-1 on 1000 dev photos, RGB vs BGR channel order."""
import numpy as np
import torch

import resnets as R

X, y = R.load_images()
_, dev, _ = R.split(y)
rng = np.random.default_rng(5)
ix = np.sort(rng.choice(dev, 1000, replace=False))
m = R.load("resnet50")
for order in ("RGB", "BGR"):
    acc = []
    with torch.no_grad():
        for b in range(0, len(ix), 100):
            xb = X[ix[b:b + 100]]
            if order == "BGR":
                xb = xb[..., ::-1]
            acc.append((R.head(m, R.last_layer(m, R.prep(xb))).argmax(1).cpu().numpy() == y[ix[b:b + 100]]))
    print(order, "top-1", np.concatenate(acc).mean() * 100)
