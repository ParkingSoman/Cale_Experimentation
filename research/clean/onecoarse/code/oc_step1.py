"""One coarse neuron, step 1: fix the triple. Parent = A stage-3 channel 69 (s3:69).
Runs on the cluster in n6/clean/neuron/coarse (needs r18.py, capture18.py, the held + test caches).

Per held + test photo (19,000):
  parent peak = max of s3:69 over the 14 x 14 grid, and its position (pr, pc)
  child position = (pr // 2, pc // 2) on the 7 x 7 grid (its 3 x 3 stride-2 window holds the parent's peak)
For each net (A, T0 = B128aug_s0_b0, B1 = B128aug_s1), block 0 of stage 4, every neuron j, at the child position:
  c69[n, j] = bn1 scale_j * sum over the window of conv1 weights (j, 69) * s3:69 values   (exact edge from the parent)
  pre[n, j] = j's input before the ReLU (bn1 output)
  pos[n, j] = max over the 7 x 7 positions of j's input before the ReLU (where it fires on the photo)
Saved: results/coarse/r18/oc_step1.npz (fp16)
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

import r18
from capture18 import net_for

torch.set_grad_enabled(False)
P = 69
TAGS = {"A": "A", "T0": "B128aug_s0_b0", "B1": "B128aug_s1"}


def main():
    nets = {k: net_for(t) for k, t in TAGS.items()}
    out = {k: [] for k in ["ppk", "parg"] + [f"{n}_{q}" for n in nets for q in ("c69", "pre", "pos")]}
    ix = []
    for s in ("held", "test"):
        cache = r18.Cache(s)
        ix.append(cache.ix)
        for a in range(0, len(cache), 250):
            x, _ = cache.get(np.arange(a, min(a + 250, len(cache))))
            n = len(x)
            xp = x[:, P]                                            # n x 14 x 14
            pk, arg = xp.flatten(1).max(1)
            pr, pc = (arg // 14).clamp(max=13), arg % 14
            cr, cc = (pr // 2).clamp(max=6), (pc // 2).clamp(max=6)
            ar = torch.arange(n, device=x.device)
            out["ppk"].append(pk.half().cpu()); out["parg"].append(arg.short().cpu())
            for name, m in nets.items():
                bl = m.layer4[0]
                sc = bl.bn1.weight / torch.sqrt(bl.bn1.running_var + bl.bn1.eps)
                pre = bl.bn1(bl.conv1(x))                           # n x k x 7 x 7
                c = F.conv2d(x[:, P:P + 1], bl.conv1.weight[:, P:P + 1], stride=2, padding=1) * sc.view(1, -1, 1, 1)
                out[f"{name}_c69"].append(c[ar, :, cr, cc].half().cpu())
                out[f"{name}_pre"].append(pre[ar, :, cr, cc].half().cpu())
                out[f"{name}_pos"].append(pre.amax((2, 3)).half().cpu())
    res = {k: torch.cat(v).numpy() for k, v in out.items()}
    ix = np.concatenate(ix)
    lab = r18.net.Photos("read").lab[ix]
    f = os.path.join(r18.OUT, "oc_step1.npz")
    np.savez(f, ix=ix, lab=lab, **res)
    print("saved", f, {k: v.shape for k, v in res.items()})


if __name__ == "__main__":
    main()
