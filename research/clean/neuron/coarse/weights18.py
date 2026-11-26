"""N11: export stage-4 weights of A and each B to numpy (results/coarse/r18/w_<tag>.npz), for reading reads and writes
on the Mac without torch. Per block b: W1_b (k x cin x 3 x 3), bn1_b (gamma, beta, mean, var), W2_b (512 x k x 3 x 3),
bn2_b; block 0 downsample (Wd 512 x 256 x 1 x 1, bnd); fc (Wfc 1000 x 512, bfc)."""
import os
import sys

import numpy as np
import torch

import r18
from capture18 import net_for

for tag in sys.argv[1:]:
    m = net_for(tag)
    out = {}
    for b, bl in enumerate(m.layer4):
        out[f"W1_{b}"] = bl.conv1.weight.detach().cpu().numpy()
        out[f"W2_{b}"] = bl.conv2.weight.detach().cpu().numpy()
        for n, bn in (("bn1", bl.bn1), ("bn2", bl.bn2)):
            out[f"{n}_{b}"] = np.stack([bn.weight.detach().cpu().numpy(), bn.bias.detach().cpu().numpy(),
                                        bn.running_mean.cpu().numpy(), bn.running_var.cpu().numpy()])
    d = m.layer4[0].downsample
    out["Wd"] = d[0].weight.detach().cpu().numpy()
    out["bnd"] = np.stack([d[1].weight.detach().cpu().numpy(), d[1].bias.detach().cpu().numpy(),
                           d[1].running_mean.cpu().numpy(), d[1].running_var.cpu().numpy()])
    out["Wfc"], out["bfc"] = m.fc.weight.detach().cpu().numpy(), m.fc.bias.detach().cpu().numpy()
    np.savez(os.path.join(r18.OUT, f"w_{tag}.npz"), **out)
    print(tag, "weights saved")
