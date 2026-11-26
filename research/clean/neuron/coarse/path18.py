"""N11: the path one layer back. For each stage-4 block-0 neuron j (of A or a B), on each of its top-50 photos
(held + test, by pos), split its pre-ReLU input at its peak position exactly into the 256 stage-3 output channels:
  pre_j = bn1 scale_j * sum_i (conv1 weights j,i  *  x_i on the 3 x 3 window)  +  bn1 shift_j
  contribution_i = bn1 scale_j * (conv1 weights j,i * x_i on the window)          (adds up to pre_j - shift_j)
x = A's frozen stage-3 output, the SAME input for A and every B. So the feeders of A's fine neurons and of B's coarse
neurons are named in one shared set of 256 stage-3 channels.
Saved: results/coarse/r18/path_<tag>.npz with feed0 [k, 256] = mean contribution over the top-50 photos, and
feed0_cars [k, 256] = the same over car photos where the neuron is above its median on car photos.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import r18
from capture18 import net_for

torch.set_grad_enabled(False)


def main():
    tag = sys.argv[1]
    m = net_for(tag)
    bl = m.layer4[0]
    d = np.load(os.path.join(r18.OUT, f"read_{tag}.npz"))
    caches = {"held": r18.Cache("held"), "test": r18.Cache("test")}
    nh = len(caches["held"])
    W = bl.conv1.weight                                     # k x 256 x 3 x 3, stride 2, padding 1
    scale = (bl.bn1.weight / torch.sqrt(bl.bn1.running_var + bl.bn1.eps))
    k = W.shape[0]

    def x_of(rows):                                         # rows into the concatenated held + test list
        rows = np.asarray(rows)
        out = []
        for r in rows:
            c, i = ("held", r) if r < nh else ("test", r - nh)
            out.append(caches[c].get([i])[0])
        return torch.cat(out)

    def contrib(j, rows):
        x = F.pad(x_of(rows), (1, 1, 1, 1))                # 256 x 16 x 16 after padding
        p = d["arg0"][rows, j].astype(int)
        r, c = p // 7, p % 7
        win = torch.stack([x[n, :, 2 * r[n]:2 * r[n] + 3, 2 * c[n]:2 * c[n] + 3] for n in range(len(rows))])
        return (scale[j] * (win * W[j][None]).sum((2, 3))).cpu().numpy()   # n x 256

    lab = d["lab"]
    cars = np.where(np.isin(lab, r18.CARS))[0]
    feed, feed_cars = np.zeros((k, 256), np.float32), np.zeros((k, 256), np.float32)
    for j in range(k):
        v = d["pos0"][:, j].astype(np.float32)
        feed[j] = contrib(j, np.argsort(-v)[:50]).mean(0)
        on = cars[v[cars] >= np.median(v[cars])]
        feed_cars[j] = contrib(j, on).mean(0)
    np.savez(os.path.join(r18.OUT, f"path_{tag}.npz"), feed0=feed, feed0_cars=feed_cars)
    print(tag, "path saved", feed.shape)


if __name__ == "__main__":
    main()
