"""One coarse neuron: how B1 n101 grows (recursive generalization). Cluster, in n6/clean/neuron/coarse.
For each checkpoint of B128aug_s1 (steps 1, 3, 10, 30, 100, 300, 1000, 3000, 6250), on the 19,000 held + test photos:
  pos[n] = n101's max over positions of (value - offset), arg[n] = that position (7 x 7)
Saved: results/coarse/r18/oc_growth.npz with pos_<step>, arg_<step>, off_<step>
"""
import copy
import os

import numpy as np
import torch

import r18

torch.set_grad_enabled(False)
STEPS = [1, 3, 10, 30, 100, 300, 1000, 3000, 6250]


def main():
    A = r18.load_a()
    held, test = r18.Cache("held"), r18.Cache("test")
    res = {}
    for st in STEPS:
        d = torch.load(os.path.join(r18.OUT, "ckpt", f"B128aug_s1_{st}.pt"), weights_only=False)
        m = r18.make_b(A, 128, 1)
        r18.load_state(m, d["state"] if "state" in d else d)
        bl = m.layer4[0]
        s = bl.bn1.weight / torch.sqrt(bl.bn1.running_var + bl.bn1.eps)
        off = float(bl.bn1.bias[101] - bl.bn1.running_mean[101] * s[101])
        pos, arg = [], []
        for c in (held, test):
            for a in range(0, len(c), 500):
                x, _ = c.get(np.arange(a, min(a + 500, len(c))))
                v = (bl.bn1(bl.conv1(x))[:, 101] - off).flatten(1)
                pos.append(v.max(1)[0].cpu()); arg.append(v.argmax(1).to(torch.uint8).cpu())
        res[f"pos_{st}"] = torch.cat(pos).numpy(); res[f"arg_{st}"] = torch.cat(arg).numpy(); res[f"off_{st}"] = off
        print("step", st, "offset", round(off, 3), "median max", round(float(np.median(res[f"pos_{st}"])), 3))
    res["ix"] = np.concatenate([held.ix, test.ix])
    np.savez(os.path.join(r18.OUT, "oc_growth.npz"), **res)


if __name__ == "__main__":
    main()
