"""V1'' (LOG): own-context unit effects. Each block-0 neuron of A (512) and B seed 1 (128), on its own 20 strongest
held + test photos, at its own peak cell per photo; step 10% of its own max; recompute block 0's write, the stream
ReLU, block 1 and the shared head exactly; change in the 1000 class scores, centred, averaged over the 20 photos
(and two halves of 10). Input: oc_effown_in.npz. Saved: results/coarse/r18/oc_effown.npz.
Runs on the cluster in n6/clean/neuron/coarse.
"""
import os
import time

import numpy as np
import torch

import r18
from capture18 import net_for

torch.set_grad_enabled(False)


def main():
    I = np.load("oc_effown_in.npz")
    held, test = r18.Cache("held"), r18.Cache("test")
    nh = len(held)

    def photos(rows):
        return torch.cat([held.get([i])[0] if i < nh else test.get([i - nh])[0] for i in rows])

    res = {}
    t0 = time.time()
    for tag, name in (("A", "A"), ("B128aug_s1", "s1")):
        m = net_for(tag); b0, b1 = m.layer4[0], m.layer4[1]
        rows, cells, steps = I[f"rows_{name}"], I[f"cells_{name}"], I[f"step_{name}"]
        k = len(steps)
        mean, h0, h1s = (np.zeros((k, 1000), np.float32) for _ in range(3))
        for j in range(k):
            x = photos(rows[j])
            inner = torch.relu(b0.bn1(b0.conv1(x))); skip = b0.downsample(x)

            def tail(inn):
                h1 = torch.relu(b0.bn2(b0.conv2(inn)) + skip)
                return r18.net.head(m, r18.net.block_parts(b1, h1)["h"])

            base = tail(inner)
            ct = torch.as_tensor(cells[j], device=x.device)
            ar = torch.arange(len(x), device=x.device)
            inn = inner.clone(); inn[ar, j, ct // 7, ct % 7] += float(steps[j])
            d = tail(inn) - base
            d = (d - d.mean(1, keepdim=True)).cpu().numpy()
            mean[j], h0[j], h1s[j] = d.mean(0), d[0::2].mean(0), d[1::2].mean(0)
            if j % 64 == 0:
                print(tag, j, f"{time.time() - t0:.0f} s", flush=True)
        res[f"{name}_mean"], res[f"{name}_h0"], res[f"{name}_h1"] = mean, h0, h1s
    f = os.path.join(r18.OUT, "oc_effown.npz")
    np.savez(f, **res)
    print("saved", f, {k: v.shape for k, v in res.items()})


if __name__ == "__main__":
    main()
