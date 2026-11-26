"""N11: read A's stage-3 output channels (the 256 stream channels after stage 3, after its ReLU) = the shared input of
A's and every B's stage 4, so the feeders ("parents") of A's fine neurons and of B's coarse neurons.
Per held + test photo (19,000, same order as capture18) and channel: pos = max over the 14 x 14 positions,
mean = mean over positions, arg = peak position (row * 14 + col). The stage-3 output is post-ReLU, so the negative
pole is read from the stage-3 last block's pre-ReLU sum z (r + skip): zmean = mean over positions of z.
Saved: results/coarse/r18/read_s3.npz
"""
import os

import numpy as np
import torch

import r18

torch.set_grad_enabled(False)


def main():
    A = r18.load_a()
    ph = r18.net.Photos("read")
    sets = r18.photo_sets(ph.lab)
    out = {k: [] for k in ("pos", "mean", "arg", "zmean")}
    ix = np.concatenate([sets["held"], sets["test"]])
    for a in range(0, len(ix), 200):
        x = ph.tensor(ix[a:a + 200]).to(r18.DEV)
        h = A.layer2(A.layer1(r18.net.stem(A, x)))
        h = A.layer3[0](h)
        p = r18.net.block_parts(A.layer3[1], h)
        hs, z = p["h"], p["z"]
        out["pos"].append(hs.amax((2, 3)).half().cpu())
        out["mean"].append(hs.mean((2, 3)).half().cpu())
        out["arg"].append(hs.flatten(2).argmax(2).to(torch.uint8).cpu())
        out["zmean"].append(z.mean((2, 3)).half().cpu())
    res = {k: torch.cat(v).numpy() for k, v in out.items()}
    np.savez(os.path.join(r18.OUT, "read_s3.npz"), ix=ix, lab=ph.lab[ix], **res)
    print("stage 3 read", {k: v.shape for k, v in res.items()})


if __name__ == "__main__":
    main()
