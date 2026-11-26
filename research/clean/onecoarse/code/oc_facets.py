"""Children the essay's way, step 6: n101's input split per photo. Runs on the cluster in n6/clean/neuron/coarse.
Per held + test photo (19,000): n101's pre-ReLU input peak over the 7 x 7 grid and its cell, and the exact
contribution of each of the 256 stage-3 channels to n101 at that cell (bn1 scale * per-channel 3 x 3 conv).
Saved: results/coarse/r18/oc_facets.npz (fp16)
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

import r18
from capture18 import net_for

torch.set_grad_enabled(False)
J = 101


def main():
    bl = net_for("B128aug_s1").layer4[0]
    sc = (bl.bn1.weight / torch.sqrt(bl.bn1.running_var + bl.bn1.eps))[J]
    w = bl.conv1.weight[J:J + 1]                                   # 1 x 256 x 3 x 3
    wg = w.view(256, 1, 3, 3)
    out = {"pk": [], "arg": [], "split": []}
    ix = []
    for s in ("held", "test"):
        cache = r18.Cache(s)
        ix.append(cache.ix)
        for a in range(0, len(cache), 250):
            x, _ = cache.get(np.arange(a, min(a + 250, len(cache))))
            n = len(x)
            pre = bl.bn1(bl.conv1(x))[:, J]                        # n x 7 x 7
            pk, arg = pre.flatten(1).max(1)
            per = F.conv2d(x, wg, stride=2, padding=1, groups=256) * sc   # n x 256 x 7 x 7
            r, c = arg // 7, arg % 7
            ar = torch.arange(n, device=x.device)
            out["pk"].append(pk.half().cpu()); out["arg"].append(arg.short().cpu())
            out["split"].append(per[ar, :, r, c].half().cpu())
    res = {k: torch.cat(v).numpy() for k, v in out.items()}
    f = os.path.join(r18.OUT, "oc_facets.npz")
    np.savez(f, ix=np.concatenate(ix), **res)
    print("saved", f, {k: v.shape for k, v in res.items()})


if __name__ == "__main__":
    main()
