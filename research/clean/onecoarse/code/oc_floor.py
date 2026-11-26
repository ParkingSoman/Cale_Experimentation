"""Floors A and B, cluster part (n6/clean/neuron/coarse).
A: for each half h of the 19,000 held + test photos (h0 = even rows, h1 = odd rows) and each A block-0 neuron j:
   feed_h[j, 256] = mean over j's top-50 photos in that half of the exact contribution of each stage-3 channel at
   j's peak cell (bn1 scale * conv1 weights on the 3 x 3 window), the same as path18.py's feed0.
B: on the train-cache photos (never shown to the reader), per A block-0 neuron j and B1 n101:
   tpos / targ = max of (value - offset) over the 7 x 7 cells and its cell; tneg / tnarg = the input-only min.
   Parent s3:69 on train photos: max of its output (after the ReLU, the cache) and its cell (14 x 14).
Saved: results/coarse/r18/oc_floor.npz
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

import r18
from capture18 import net_for

torch.set_grad_enabled(False)


def main():
    A, B1 = net_for("A"), net_for("B128aug_s1")
    a0, c0 = A.layer4[0], B1.layer4[0]
    sA = a0.bn1.weight / torch.sqrt(a0.bn1.running_var + a0.bn1.eps); tA = a0.bn1.bias - a0.bn1.running_mean * sA
    sC = c0.bn1.weight / torch.sqrt(c0.bn1.running_var + c0.bn1.eps); tC = c0.bn1.bias - c0.bn1.running_mean * sC
    d = np.load(os.path.join(r18.OUT, "read_A.npz"))
    held, test = r18.Cache("held"), r18.Cache("test")
    nh = len(held)
    assert (d["ix"] == np.concatenate([held.ix, test.ix])).all()
    pos0, arg0 = d["pos0"].astype(np.float32), d["arg0"].astype(int)

    def x_of(rows):
        rows = np.sort(rows)
        return rows, torch.cat([held.get(rows[rows < nh])[0], test.get(rows[rows >= nh] - nh)[0]])

    res = {}
    W = a0.conv1.weight
    for h in (0, 1):
        idx = np.arange(h, len(pos0), 2)
        feed = np.zeros((512, 256), np.float32)
        for j in range(512):
            rows = idx[np.argsort(-pos0[idx, j])[:50]]
            rows, x = x_of(rows)
            xp = F.pad(x, (1, 1, 1, 1))
            p = arg0[rows, j]; r, c = p // 7, p % 7
            win = torch.stack([xp[n, :, 2 * r[n]:2 * r[n] + 3, 2 * c[n]:2 * c[n] + 3] for n in range(len(rows))])
            feed[j] = (sA[j] * (win * W[j][None]).sum((2, 3))).mean(0).cpu().numpy()
        res[f"feed_h{h}"] = feed
        print("half", h, "done")
    tr = r18.Cache("train")
    out = {k: [] for k in ("tpos", "targ", "tneg", "tnarg", "cpos", "carg", "cneg", "cnarg", "ppos", "parg")}
    for a in range(0, len(tr), 500):
        x, _ = tr.get(np.arange(a, min(a + 500, len(tr))))
        v = (a0.bn1(a0.conv1(x)) - tA.view(1, -1, 1, 1)).flatten(2)
        out["tpos"].append(v.max(2)[0].half().cpu()); out["targ"].append(v.argmax(2).to(torch.uint8).cpu())
        out["tneg"].append(v.min(2)[0].half().cpu()); out["tnarg"].append(v.argmin(2).to(torch.uint8).cpu())
        w = (c0.bn1(c0.conv1(x))[:, 101] - tC[101]).flatten(1)
        out["cpos"].append(w.max(1)[0].cpu()); out["carg"].append(w.argmax(1).to(torch.uint8).cpu())
        out["cneg"].append(w.min(1)[0].cpu()); out["cnarg"].append(w.argmin(1).to(torch.uint8).cpu())
        q = x[:, 69].flatten(1)
        out["ppos"].append(q.max(1)[0].cpu()); out["parg"].append(q.argmax(1).to(torch.uint8).cpu())
    for k, v in out.items():
        res[k] = torch.cat(v).numpy()
    res["train_ix"] = tr.ix
    np.savez(os.path.join(r18.OUT, "oc_floor.npz"), **res)
    print("saved", {k: v.shape for k, v in res.items()})


if __name__ == "__main__":
    main()
