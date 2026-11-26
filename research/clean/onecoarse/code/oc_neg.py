"""One coarse neuron: the negative pole with the same tool as the positive pole (README reading rule 1).
Cluster, in n6/clean/neuron/coarse. Per held + test photo (19,000, the caches' order) and per neuron of the triple:
  neg = min over positions of the input before the ReLU, nar = the position of that min (row * grid + col)
Parent s3:69: A's stage-3 output is after a ReLU, so the parent is read on the input of that ReLU
(z = layer3[1]'s residual sum, 14 x 14), on both sides: p_pos / p_arg (max) and p_neg / p_nar (min).
Children (A block 0) and the coarse neuron (B1 block 0): from the cached stage-3 outputs, 7 x 7 grid.
Saved: results/coarse/r18/oc_neg.npz
"""
import os

import numpy as np
import torch

import r18
from capture18 import net_for

torch.set_grad_enabled(False)
KIDS = [305, 97, 23, 297, 108, 163]


def main():
    A, B1 = net_for("A"), net_for("B128aug_s1")
    ph = r18.net.Photos("read")
    out = {k: [] for k in ("p_pos", "p_arg", "p_neg", "p_nar", "k_neg", "k_nar", "c_neg", "c_nar")}
    ixs = []
    for s in ("held", "test"):
        cache = r18.Cache(s)
        ixs.append(cache.ix)
        for a in range(0, len(cache), 250):
            rows = np.arange(a, min(a + 250, len(cache)))
            x, _ = cache.get(rows)
            img = ph.tensor(cache.ix[rows]).to(r18.DEV)
            h = A.layer3[0](A.layer2(A.layer1(r18.net.stem(A, img))))
            z = r18.net.block_parts(A.layer3[1], h)["z"][:, 69].flatten(1)
            if a == 0 and s == "held":
                hs = torch.relu(z).amax(1)
                print("check: relu(z) max vs cached x max, max abs diff", float((hs - x[:, 69].flatten(1).amax(1)).abs().max()))
            out["p_pos"].append(z.max(1)[0].half().cpu()); out["p_arg"].append(z.argmax(1).to(torch.uint8).cpu())
            out["p_neg"].append(z.min(1)[0].half().cpu()); out["p_nar"].append(z.argmin(1).to(torch.uint8).cpu())
            for key, m, js in (("k", A, KIDS), ("c", B1, [101])):
                bl = m.layer4[0]
                pre = bl.bn1(bl.conv1(x))[:, js].flatten(2)
                out[f"{key}_neg"].append(pre.min(2)[0].half().cpu()); out[f"{key}_nar"].append(pre.argmin(2).to(torch.uint8).cpu())
    res = {k: torch.cat(v).numpy() for k, v in out.items()}
    ix = np.concatenate(ixs)
    np.savez(os.path.join(r18.OUT, "oc_neg.npz"), ix=ix, lab=ph.lab[ix], kids=np.array(KIDS), **res)
    print({k: v.shape for k, v in res.items()})


if __name__ == "__main__":
    main()
