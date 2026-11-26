"""V1'd (LOG): unit effects of all 512 block-0 neurons of B512aug_s0 (full-width B), same method and photos as
oc_effall.py (centre cell, 100 photos, step 10% of own max). Input: oc_eff512_in.npz.
Saved: results/coarse/r18/oc_eff512.npz. Runs on the cluster in n6/clean/neuron/coarse.
"""
import os

import numpy as np
import torch

import r18
from capture18 import net_for

torch.set_grad_enabled(False)
CELL = 24


def main():
    I = np.load("oc_eff512_in.npz")
    rows = I["rows"]
    held, test = r18.Cache("held"), r18.Cache("test")
    nh = len(held)
    x = torch.cat([held.get([i])[0] if i < nh else test.get([i - nh])[0] for i in rows])
    r, c = CELL // 7, CELL % 7
    m = net_for("B512aug_s0"); b0, b1 = m.layer4[0], m.layer4[1]
    inner = torch.relu(b0.bn1(b0.conv1(x))); skip = b0.downsample(x)

    def tail(inn):
        h1 = torch.relu(b0.bn2(b0.conv2(inn)) + skip)
        return r18.net.head(m, r18.net.block_parts(b1, h1)["h"])

    base = tail(inner)
    k = inner.shape[1]
    mean, h0, h1s = (np.zeros((k, 1000), np.float32) for _ in range(3))
    for j in range(k):
        inn = inner.clone(); inn[:, j, r, c] += float(I["step"][j])
        d = tail(inn) - base
        d = (d - d.mean(1, keepdim=True)).cpu().numpy()
        mean[j], h0[j], h1s[j] = d.mean(0), d[0::2].mean(0), d[1::2].mean(0)
    f = os.path.join(r18.OUT, "oc_eff512.npz")
    np.savez(f, rows=rows, b512_mean=mean, b512_h0=h0, b512_h1=h1s)
    print("saved", f, mean.shape)


if __name__ == "__main__":
    main()
