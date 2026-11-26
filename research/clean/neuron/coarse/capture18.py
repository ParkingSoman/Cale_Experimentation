"""N11 reading capture on ResNet-18 stage 4, for A or a trained B.

For every held and test photo (19,000; never train), and every inner neuron of stage-4 blocks 0 and 1:
  pos[n, j]  = max over positions of the neuron's input before the ReLU (bn1 output). Positive pole: where it fires.
  mean[n, j] = mean over positions of the same input. Negative pole: where it is pushed down hardest.
  arg[n, j]  = the position of the peak on the 7 x 7 grid (row * 7 + col), for crops centred on it.
Saved: results/coarse/r18/read_<tag>.npz with pos0, mean0, pos1, mean1 (fp16), ix (photo index), lab (class).
tag = "A" or a B tag such as "B_k32_s0" (loads its best state).
"""
import os
import sys

import numpy as np
import torch

import r18

torch.set_grad_enabled(False)


def net_for(tag):
    A = r18.load_a()
    if tag == "A":
        return A
    d = torch.load(os.path.join(r18.OUT, f"{tag}.pt"), weights_only=False)
    m = r18.make_b(A, d["k"], d["seed"], d.get("blocks", (0, 1)))
    r18.load_state(m, d["state"])
    return m.eval()


def capture(m, cache):
    out = {k: [] for k in ("pos0", "mean0", "pos1", "mean1", "arg0", "arg1")}
    for a in range(0, len(cache), 250):
        x, _ = cache.get(np.arange(a, min(a + 250, len(cache))))
        h = x
        for b, bl in enumerate(m.layer4):
            pre = bl.bn1(bl.conv1(h))
            out[f"pos{b}"].append(pre.amax((2, 3)).half().cpu())
            out[f"arg{b}"].append(pre.flatten(2).argmax(2).to(torch.uint8).cpu())     # peak position, row * 7 + col
            out[f"mean{b}"].append(pre.mean((2, 3)).half().cpu())
            h = r18.net.block_parts(bl, h)["h"]
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def main():
    tag = sys.argv[1]
    m = net_for(tag)
    lab = r18.net.Photos("read").lab
    parts = [capture(m, r18.Cache(s)) for s in ("held", "test")]
    ix = np.concatenate([r18.Cache(s).ix for s in ("held", "test")])
    res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez(os.path.join(r18.OUT, f"read_{tag}.npz"), ix=ix, lab=lab[ix], **res)
    print(tag, {k: v.shape for k, v in res.items()})


if __name__ == "__main__":
    main()
