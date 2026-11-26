"""N11 exact edges in ResNet-18 stage 4, on car photos (held + test cars, never train).

E1[j, c] = mean over car photos of  pos_c(full) - pos_c(parent j filled with its car-photo mean at every position)
  parent j = a stage-4 block-0 inner neuron, child c = a block-1 inner neuron, pos = max over positions of the
  child's pre-ReLU input. Exact: block 0's conv2, bn2, the stream ReLU, block 1's conv1 and bn1 all recomputed.
E0[i, j] = the same with parent i = a stage-3 output stream channel (filled with its car-photo mean, which also
  changes the downsample skip) and child j = a block-0 inner neuron.
Also D1[c, :] = each block-1 neuron's exact effect on the 11 car logits (logits(full) - logits(c filled with its
  car mean)), mean over car photos, and D0[j, :] the same for block-0 neurons.
Saved: results/coarse/r18/edges_<tag>.npz
"""
import os
import sys

import numpy as np
import torch

import r18
from capture18 import net_for

torch.set_grad_enabled(False)


def car_x():
    xs = []
    for s in ("held", "test"):
        c = r18.Cache(s)
        lab = r18.net.Photos("read").lab[c.ix]
        r = np.where(np.isin(lab, r18.CARS))[0]
        xs.append(c.get(r)[0])
    return torch.cat(xs)


def main():
    tag = sys.argv[1]
    m = net_for(tag)
    b0, b1 = m.layer4[0], m.layer4[1]
    x = car_x()
    ci = torch.as_tensor(r18.CARS, device=r18.DEV)

    def fwd(x, fill0=None, fill_s3=None, fill1=None):
        if fill_s3 is not None:
            i, v = fill_s3
            x = x.clone(); x[:, i] = v
        pre0 = b0.bn1(b0.conv1(x))
        inner0 = torch.relu(pre0)
        if fill0 is not None:
            j, v = fill0
            inner0 = inner0.clone(); inner0[:, j] = v
        h1 = torch.relu(b0.bn2(b0.conv2(inner0)) + b0.downsample(x))
        pre1 = b1.bn1(b1.conv1(h1))
        inner1 = torch.relu(pre1)
        if fill1 is not None:
            c, v = fill1
            inner1 = inner1.clone(); inner1[:, c] = v
        h2 = torch.relu(b1.bn2(b1.conv2(inner1)) + h1)
        return pre0, inner0, pre1, inner1, r18.net.head(m, h2)[:, ci]

    pre0, inner0, pre1, inner1, lg = fwd(x)
    k0, k1 = inner0.shape[1], inner1.shape[1]
    m_in0 = inner0.mean((0, 2, 3)); m_in1 = inner1.mean((0, 2, 3)); m_x = x.mean((0, 2, 3))
    pos1, pos0 = pre1.amax((2, 3)), pre0.amax((2, 3))
    E1 = np.zeros((k0, k1), np.float32); D0 = np.zeros((k0, 11), np.float32)
    for j in range(k0):
        _, _, p1, _, l = fwd(x, fill0=(j, m_in0[j]))
        E1[j] = (pos1 - p1.amax((2, 3))).mean(0).cpu().numpy()
        D0[j] = (lg - l).mean(0).cpu().numpy()
    E0 = np.zeros((x.shape[1], k0), np.float32)
    for i in range(x.shape[1]):
        p0, _, _, _, _ = fwd(x, fill_s3=(i, m_x[i]))
        E0[i] = (pos0 - p0.amax((2, 3))).mean(0).cpu().numpy()
    D1 = np.zeros((k1, 11), np.float32)
    for c in range(k1):
        *_, l = fwd(x, fill1=(c, m_in1[c]))
        D1[c] = (lg - l).mean(0).cpu().numpy()
    np.savez(os.path.join(r18.OUT, f"edges_{tag}.npz"), E1=E1, E0=E0, D1=D1, D0=D0, n_photos=len(x),
             pos1_std=pos1.std(0).cpu().numpy(), pos0_std=pos0.std(0).cpu().numpy())
    print(tag, "car photos", len(x), "E1", E1.shape, "E0", E0.shape)


if __name__ == "__main__":
    main()
