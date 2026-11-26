"""Fair check for A3 (absorb2.py): give the coarse children the same KL-best bias that A1 and A2 got.
Also prints each child's baseline pre-activation (all s4b0 inputs at their means): if it is above 0, the coarse child
fires on every photo, and a constant shift follows."""
import json
import os

import numpy as np
import torch

from absorb import net, OUT
from controls import solve, boot
from fit import kl_rows


def main():
    cache = np.load(os.path.join(OUT, "absorb_cache.npz"))
    feats = np.load(os.path.join(OUT, "absorb2_cache.npz"))
    lab, fit = cache["lab"], cache["fit"]
    W, bias = torch.tensor(cache["fcW"]).double(), torch.tensor(cache["fcb"]).double()
    yN = torch.tensor(cache["gN"]).double() @ W.T + bias
    F, T = np.where(fit)[0], np.where(~fit)[0]
    res = []
    for p in (245, 418):
        row = dict(parent=p, goal="all")
        yD = torch.tensor(feats[f"{p}_drop"]).double() @ W.T + bias
        a, _ = solve(yN[F], yD[F], None)
        row["drop"], row["drop_bias"] = float(kl_rows(yN[T], yD[T]).mean()), float(kl_rows(yN[T], yD[T] + a).mean())
        for v in ("A3r", "A3all"):
            y = torch.tensor(feats[f"{p}_{v}"]).double() @ W.T + bias
            a, _ = solve(yN[F], y[F], None)
            row[v] = float(kl_rows(yN[T], y[T]).mean())
            row[v + "_bias"] = float(kl_rows(yN[T], y[T] + a).mean())
        res.append(row)
        print(json.dumps(row), flush=True)
    torch.set_default_dtype(torch.float32)
    m = net.load()
    h_mean = net.block_means(os.path.join(net.STORE, "read"))[4][0]
    b1 = m.layer4[1]
    with torch.no_grad():
        u_fill = b1.bn1(b1.conv1(h_mean.view(1, -1, 1, 1).expand(1, 512, 7, 7).contiguous()))[0]
    base = {j: float(u_fill[j, 3, 3]) for j in (243, 73, 377, 317, 408)}
    print("baseline pre-activation at the centre:", base, flush=True)
    json.dump(dict(rows=res, baseline=base), open(os.path.join(OUT, "a3bias.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
