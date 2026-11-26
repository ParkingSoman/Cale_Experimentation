"""V1' (LOG "Writes: a translation through the readers"): unit effects of ALL block-0 neurons of A (512),
B seed 0 (128) and B seed 1 (128), at the centre cell (3, 3), on 100 random held + test photos.
Unit effect = add 10% of the neuron's own max to its output (after the ReLU) at the cell, recompute block 0's write,
the stream ReLU, block 1 and the shared head exactly; change in the 1000 class scores, centred over classes.
Input: oc_effall_in.npz. Saved: results/coarse/r18/oc_effall.npz (mean over photos, and the two half means).
Runs on the cluster in n6/clean/neuron/coarse.
"""
import os
import time

import numpy as np
import torch

import r18
from capture18 import net_for

torch.set_grad_enabled(False)
CELL = 24


def main():
    I = np.load("oc_effall_in.npz")
    rows = I["rows"]
    held, test = r18.Cache("held"), r18.Cache("test")
    nh = len(held)
    x = torch.cat([held.get([i])[0] if i < nh else test.get([i - nh])[0] for i in rows])
    r, c = CELL // 7, CELL % 7
    res = {}
    t0 = time.time()
    for tag, key in (("A", "step_A"), ("B128aug_s0", "step_s0"), ("B128aug_s1", "step_s1")):
        m = net_for(tag); b0, b1 = m.layer4[0], m.layer4[1]
        inner = torch.relu(b0.bn1(b0.conv1(x))); skip = b0.downsample(x)

        def tail(inn):
            h1 = torch.relu(b0.bn2(b0.conv2(inn)) + skip)
            return r18.net.head(m, r18.net.block_parts(b1, h1)["h"])

        base = tail(inner)
        k = inner.shape[1]
        mean, h0, h1s = np.zeros((k, 1000), np.float32), np.zeros((k, 1000), np.float32), np.zeros((k, 1000), np.float32)
        for j in range(k):
            inn = inner.clone(); inn[:, j, r, c] += float(I[key][j])
            d = tail(inn) - base
            d = (d - d.mean(1, keepdim=True)).cpu().numpy()
            mean[j], h0[j], h1s[j] = d.mean(0), d[0::2].mean(0), d[1::2].mean(0)
        name = {"A": "A", "B128aug_s0": "s0", "B128aug_s1": "s1"}[tag]
        res[f"{name}_mean"], res[f"{name}_h0"], res[f"{name}_h1"] = mean, h0, h1s
        print(tag, "done", f"{time.time() - t0:.0f} s", flush=True)
    f = os.path.join(r18.OUT, "oc_effall.npz")
    np.savez(f, rows=rows, **res)
    print("saved", f, {k: v.shape for k, v in res.items()})


if __name__ == "__main__":
    main()
