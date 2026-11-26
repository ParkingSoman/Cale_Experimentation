# scale: neurons=s3:69, A:children, 41 stage-3 channels (photo groups only); net=A; photos=held; where=cluster
# rung: Q=C1 is coarsening local to a capability? stage-3 channel peaks on DEV; N=1; EST=2 min CPU; PREV=results/oc_dial.npz; PERMIT=Shonu approved plan "Should coarsening keep the parent and drop the children's other reads?" 2026-11-24 incl. its SCALE-UP REQUEST
"""C1 data (LOG "Should coarsening drop the children's other reads?").

Saves, for the 2,000 held (DEV) photos: the peak of every stage-3 output channel (photos x 256), and the
positive read weight of each of the 11 R1 children on each stage-3 channel (11 x 256, bn1 scale in, summed over the
3 x 3 kernel, negatives set to 0). The KL of delete and morph per DEV photo is already in oc_dial.npz (set11 rows).
The bar is read on the Mac (oc_capability_read.py).
Usage: python -u oc_capability.py
"""
import os
import time

import numpy as np
import torch

import r18
from capture18 import net_for

torch.set_grad_enabled(False)
P, K_SET = 69, 11


def main():
    t0 = time.time()
    m = net_for("A").to(r18.DEV).eval()
    b0 = m.layer4[0]
    sc = b0.bn1.weight / torch.sqrt(b0.bn1.running_var + b0.bn1.eps)
    R = b0.conv1.weight * sc.view(-1, 1, 1, 1)                                    # 512 x 256 x 3 x 3
    V = b0.conv2.weight * (b0.bn2.weight / torch.sqrt(b0.bn2.running_var + b0.bn2.eps)).view(-1, 1, 1, 1)
    score = (R[:, P].sum((1, 2)) * torch.sqrt((V ** 2).sum((0, 2, 3)))).cpu().numpy()
    order = np.argsort(-score)
    kids = order[:K_SET].copy()
    wpos = torch.clamp(R[torch.as_tensor(kids)].sum((2, 3)), min=0).cpu().numpy()   # 11 x 256
    print("kids:", kids.tolist(), flush=True)

    held = r18.Cache("held")
    rows = np.arange(len(held))
    pk = []
    for s in range(0, len(rows), 250):
        x, _ = held.get(rows[s:s + 250])
        pk.append(x.float().amax((2, 3)).cpu().numpy())
    pk = np.concatenate(pk)                                                       # 2000 x 256
    out = os.path.join(r18.OUT, "oc_capability.npz")
    np.savez(out, order=order, kids=kids, wpos=wpos, dev_rows=rows, s3_peak=pk)
    print(f"saved {out}; peak shape {pk.shape}; parent peak mean {pk[:, P].mean():.3f}; {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
