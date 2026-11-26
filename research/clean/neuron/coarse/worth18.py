"""N11: each stage-4 neuron's worth for the car family, ResNet-18 (A or a B), on car photos (held + test cars).
Fill = the neuron's mean over car photos, at every position (the same fill as edge18.py).
  split[j] = mean over car photos of  log q(true car | 11 cars)  full  minus  filled.   > 0: j helps pick the right car
  mass[j]  = mean over car photos of  log q(any of the 11 cars)  full  minus  filled.   > 0: j holds up "car"
  perclass[j, c] = split worth on the photos of car c only.
Also for stage-3 output stream channels (the inputs of stage 4): split3, mass3, perclass3.
Saved: results/coarse/r18/worth_<tag>.npz
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import r18
from capture18 import net_for

torch.set_grad_enabled(False)


def car_xy():
    xs, ys = [], []
    lab_all = r18.net.Photos("read").lab
    for s in ("held", "test"):
        c = r18.Cache(s)
        lab = lab_all[c.ix]
        r = np.where(np.isin(lab, r18.CARS))[0]
        xs.append(c.get(r)[0]); ys.append(lab[r])
    return torch.cat(xs), np.concatenate(ys)


def main():
    tag = sys.argv[1]
    m = net_for(tag)
    b0, b1 = m.layer4[0], m.layer4[1]
    x, y = car_xy()
    yi = torch.as_tensor([r18.CARS.index(int(v)) for v in y], device=r18.DEV)
    ci = torch.as_tensor(r18.CARS, device=r18.DEV)

    def fwd(x, f3=None, f0=None, f1=None):
        if f3 is not None:
            x = x.clone(); x[:, f3[0]] = f3[1]
        in0 = torch.relu(b0.bn1(b0.conv1(x)))
        if f0 is not None:
            in0 = in0.clone(); in0[:, f0[0]] = f0[1]
        h1 = torch.relu(b0.bn2(b0.conv2(in0)) + b0.downsample(x))
        in1 = torch.relu(b1.bn1(b1.conv1(h1)))
        if f1 is not None:
            in1 = in1.clone(); in1[:, f1[0]] = f1[1]
        lg = r18.net.head(m, torch.relu(b1.bn2(b1.conv2(in1)) + h1))
        ls = F.log_softmax(lg, 1)
        true = F.log_softmax(lg[:, ci], 1).gather(1, yi[:, None])[:, 0]
        mass = torch.logsumexp(ls[:, ci], 1)
        return true, mass, in0, in1

    t0, m0, in0, in1 = fwd(x)
    mu0, mu1, mux = in0.mean((0, 2, 3)), in1.mean((0, 2, 3)), x.mean((0, 2, 3))
    out = {}
    for name, k, kw in (("0", in0.shape[1], "f0"), ("1", in1.shape[1], "f1"), ("3", x.shape[1], "f3")):
        mu = {"f0": mu0, "f1": mu1, "f3": mux}[kw]
        sp, ms, pc = np.zeros(k, np.float32), np.zeros(k, np.float32), np.zeros((k, 11), np.float32)
        for j in range(k):
            t, mm, _, _ = fwd(x, **{kw: (j, mu[j])})
            d = (t0 - t).cpu().numpy()
            sp[j], ms[j] = d.mean(), float((m0 - mm).mean())
            pc[j] = [d[yi.cpu().numpy() == c].mean() for c in range(11)]
        out[f"split{name}"], out[f"mass{name}"], out[f"perclass{name}"] = sp, ms, pc
    np.savez(os.path.join(r18.OUT, f"worth_{tag}.npz"), n_photos=len(x), **out)
    print(tag, len(x), "car photos;", {k: float(np.sort(v)[-1]) for k, v in out.items() if v.ndim == 1})


if __name__ == "__main__":
    main()
