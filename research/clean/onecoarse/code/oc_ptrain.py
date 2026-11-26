"""Parent s3:69 on the train-cache photos (the blind-test photos): z = layer3[1] residual sum before the ReLU, minus
its bn2 shift (input only). Saves its max, min and their cells (14 x 14). -> results/coarse/r18/oc_ptrain.npz"""
import os

import numpy as np
import torch

import r18

torch.set_grad_enabled(False)
A = r18.load_a(); ph = r18.net.Photos("read"); tr = r18.Cache("train")
b = A.layer3[1]
s2 = b.bn2.weight / torch.sqrt(b.bn2.running_var + b.bn2.eps); off = float(b.bn2.bias[69] - b.bn2.running_mean[69] * s2[69])
out = {k: [] for k in ("zmax", "zarg", "zmin", "znarg")}
for a in range(0, len(tr), 250):
    rows = np.arange(a, min(a + 250, len(tr)))
    img = ph.tensor(tr.ix[rows]).to(r18.DEV)
    h = A.layer3[0](A.layer2(A.layer1(r18.net.stem(A, img))))
    z = (r18.net.block_parts(b, h)["z"][:, 69] - off).flatten(1)
    out["zmax"].append(z.max(1)[0].cpu()); out["zarg"].append(z.argmax(1).to(torch.uint8).cpu())
    out["zmin"].append(z.min(1)[0].cpu()); out["znarg"].append(z.argmin(1).to(torch.uint8).cpu())
np.savez(os.path.join(r18.OUT, "oc_ptrain.npz"), train_ix=tr.ix, **{k: torch.cat(v).numpy() for k, v in out.items()})
print("saved", len(tr))
