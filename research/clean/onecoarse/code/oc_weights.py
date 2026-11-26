"""Dump the weights of the one-coarse system (cluster, n6/clean/neuron/coarse).
reads: bn1-scaled conv1 rows [k, 256, 3, 3] in stage-3 channel space, plus the bn1 offset.
writes: bn2-scaled conv2 columns [512, k, 3, 3] into the 512 stream channels.
A b0: the 7 core/ring children used here; B1 b0: n101. Saved: results/coarse/r18/oc_weights.npz"""
import os

import numpy as np
import torch

import r18
from capture18 import net_for

KIDS = [305, 97, 23, 297, 108, 163, 275]


def parts(bl, js):
    s1 = bl.bn1.weight / torch.sqrt(bl.bn1.running_var + bl.bn1.eps); t1 = bl.bn1.bias - bl.bn1.running_mean * s1
    s2 = bl.bn2.weight / torch.sqrt(bl.bn2.running_var + bl.bn2.eps)
    read = (bl.conv1.weight[js] * s1[js].view(-1, 1, 1, 1)).detach().cpu().numpy()
    write = (bl.conv2.weight[:, js] * s2.view(-1, 1, 1, 1)).detach().cpu().numpy()
    return read, t1[js].detach().cpu().numpy(), write


A, B1 = net_for("A"), net_for("B128aug_s1")
ra, oa, wa = parts(A.layer4[0], KIDS)
rc, oc, wc = parts(B1.layer4[0], [101])
allA, _, _ = parts(A.layer4[0], list(range(512)))
np.savez(os.path.join(r18.OUT, "oc_weights.npz"), kids=np.array(KIDS), read_kids=ra, off_kids=oa, write_kids=wa,
         read_c101=rc, off_c101=oc, write_c101=wc, read_all_A=allA)
print("saved", ra.shape, wa.shape, rc.shape, allA.shape)
