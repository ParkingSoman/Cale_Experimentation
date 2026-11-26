"""Writes: a translation through the readers (LOG, same name). Runs on the cluster in n6/clean/neuron/coarse.
Input: oc_effect_in.npz (rows into held + test, n101's peak cell per photo, step sizes = 10% of each neuron's max).
For each photo and each scored neuron: add the step to the neuron's output (after the ReLU) at n101's peak cell,
recompute block 0's write, the stream ReLU, block 1 and the shared head exactly, and record the change in the
1000 class scores (centred over classes). Also the neuron's own activation at that cell.
Neurons: B seed 1 n101; all 128 B seed 0 block-0 neurons; the A children in `kids`.
Saved: results/coarse/r18/oc_effect.npz (fp16). Usage: python oc_effect.py [n_photos]
"""
import os
import sys
import time

import numpy as np
import torch

import r18
from capture18 import net_for

torch.set_grad_enabled(False)


def head(m, h):
    return r18.net.head(m, h)


def unit_effects(m, x, cells, js, steps):
    b0, b1 = m.layer4[0], m.layer4[1]
    inner = torch.relu(b0.bn1(b0.conv1(x)))
    skip = b0.downsample(x)

    def tail(inn):
        h1 = torch.relu(b0.bn2(b0.conv2(inn)) + skip)
        h2 = r18.net.block_parts(b1, h1)["h"]
        return head(m, h2)

    base = tail(inner)
    n = len(x); ar = torch.arange(n, device=x.device)
    r, c = cells // 7, cells % 7
    eff, act = [], []
    for j, s in zip(js, steps):
        inn = inner.clone()
        inn[ar, j, r, c] += float(s)
        d = tail(inn) - base
        eff.append((d - d.mean(1, keepdim=True)).half().cpu())
        act.append(inner[ar, j, r, c].half().cpu())
    return torch.stack(eff, 1), torch.stack(act, 1)                 # n x J x 1000, n x J


def main():
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else None
    I = np.load("oc_effect_in.npz")
    rows, cells = I["rows"], I["cells"]
    if nmax:
        rows, cells = rows[:nmax], cells[:nmax]
    held, test = r18.Cache("held"), r18.Cache("test")
    nh = len(held)
    nets = {"s1": (net_for("B128aug_s1"), [101], I["step_s1"]),
            "s0": (net_for("B128aug_s0"), list(range(128)), I["step_s0"]),
            "A": (net_for("A"), [int(k) for k in I["kids"]], I["step_A"])}
    out = {f"{k}_{q}": [] for k in nets for q in ("eff", "act")}
    t0 = time.time()
    for a in range(0, len(rows), 50):
        rr, cc = rows[a:a + 50], cells[a:a + 50]
        xs = [held.get([i])[0] if i < nh else test.get([i - nh])[0] for i in rr]
        x = torch.cat(xs)
        ct = torch.as_tensor(cc, device=x.device)
        for k, (m, js, steps) in nets.items():
            e, ac = unit_effects(m, x, ct, js, steps)
            out[f"{k}_eff"].append(e); out[f"{k}_act"].append(ac)
        print(f"{a + len(rr)} photos, {time.time() - t0:.0f} s", flush=True)
    res = {k: torch.cat(v).numpy() for k, v in out.items()}
    f = os.path.join(r18.OUT, "oc_effect_smoke.npz" if nmax else "oc_effect.npz")
    np.savez(f, rows=rows, cells=cells, kids=I["kids"], **res)
    print("saved", f, {k: v.shape for k, v in res.items()})
    for k in nets:
        e = res[f"{k}_eff"].astype(np.float32)
        print(k, "mean |effect|", float(np.abs(e).mean()), "zero rows", int((np.abs(e).sum(2) == 0).sum()))


if __name__ == "__main__":
    main()
