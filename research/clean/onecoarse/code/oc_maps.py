"""One coarse neuron: whole activation maps on the photos shown on the band sheets (both poles in one photo).
Cluster, in n6/clean/neuron/coarse. Input: oc_map_rows.npy (rows into the held + test order of the caches).
Saved: results/coarse/r18/oc_maps.npz with
  pz   [n, 14, 14]  parent s3:69 input of its ReLU (layer3[1] residual sum z)
  kids [n, 6, 7, 7] A block-0 children's inputs before the ReLU (305, 97, 23, 297, 108, 163)
  c    [n, 7, 7]    B1 block-0 n101's input before the ReLU
"""
import os

import numpy as np
import torch

import r18
from capture18 import net_for

torch.set_grad_enabled(False)
KIDS = [305, 97, 23, 297, 108, 163]


def main():
    rows = np.load("oc_map_rows.npy")
    A, B1 = net_for("A"), net_for("B128aug_s1")
    ph = r18.net.Photos("read")
    held, test = r18.Cache("held"), r18.Cache("test")
    nh = len(held)
    ix = np.concatenate([held.ix, test.ix])[rows]
    img = ph.tensor(ix).to(r18.DEV)
    h = A.layer3[0](A.layer2(A.layer1(r18.net.stem(A, img))))
    p = r18.net.block_parts(A.layer3[1], h)
    x = p["h"]
    xc = torch.cat([held.get(rows[rows < nh])[0], test.get(rows[rows >= nh] - nh)[0]])
    print("check: recomputed stage-3 output vs cache, max abs diff", float((x - xc).abs().max()))
    kids = A.layer4[0].bn1(A.layer4[0].conv1(x))[:, KIDS]
    c = B1.layer4[0].bn1(B1.layer4[0].conv1(x))[:, 101]
    np.savez(os.path.join(r18.OUT, "oc_maps.npz"), rows=rows, ix=ix, pz=p["z"][:, 69].cpu().numpy(),
             kids=kids.cpu().numpy(), c=c.cpu().numpy())
    print("saved", len(rows))


if __name__ == "__main__":
    main()
