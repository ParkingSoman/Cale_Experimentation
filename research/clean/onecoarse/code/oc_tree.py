# scale: neurons=s3:69 + A block-0 (512) + B block-0 (128); net=A,B; photos=4,000 test; where=mac
# rung: Q=CALC 3.7 the actual tree (parent, R1 children, coarse n101); EST=3 min Mac; PERMIT=Shonu approved plan 2026-11-25
"""CALC 3.7 data (distill/CALC.md 3.7, pre-registered). One Mac torch job.

Photos for reading only: 4,000 test photos (odd positions, the first 4 per class; labels unused).
Stage-3 output x = upto(A, photo, 4) (A = torchvision ResNet-18). Then, from x only:
  A_pre  (N, 512, 49): A stage-4 block-0 bn1(conv1(x)), all 512 neurons, 7 x 7 cells
  B_pre  (N, 128, 49): B128aug_s1 block-0 bn1(conv1(x)) from w_B128aug_s1.npz
  pwin   (N, 49):      max of s3:69 (x channel 69, after the ReLU) over each cell's 3 x 3 window
Checks: A_pre from the torch model = A_pre from w_A.npz; changing x at (2r, 2c) moves only cell (r, c).
Output: onecoarse/results/oc_tree.npz
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(CLEAN, "curves", "map_agents", "resnet18"))
import net  # noqa: E402

torch.set_grad_enabled(False)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
WD = os.path.join(CLEAN, "neuron", "results", "coarse", "r18")


def conv_bn(x, W, bn):
    g, b, m, v = [torch.tensor(t, dtype=torch.float32, device=x.device) for t in bn]
    y = F.conv2d(x, torch.tensor(W, dtype=torch.float32, device=x.device), stride=2, padding=1)
    return (y - m[None, :, None, None]) / torch.sqrt(v[None, :, None, None] + 1e-5) * g[None, :, None, None] + b[None, :, None, None]


def main():
    smoke = "--smoke" in sys.argv
    ph = net.Photos("read")
    ids = []
    for c in range(1000 if not smoke else 10):
        w = np.where(ph.lab == c)[0]
        ids += w[1::2][:4].tolist()
    ids = np.array(ids)
    A = net.load().to(DEV)
    wA, wB = np.load(os.path.join(WD, "w_A.npz")), np.load(os.path.join(WD, "w_B128aug_s1.npz"))
    a0 = A.layer4[0]
    Ap, Bp, PW = [], [], []
    for i in range(0, len(ids), 100):
        x = net.upto(A, ph.tensor(ids[i:i + 100]).to(DEV), 4)
        ya = a0.bn1(a0.conv1(x))
        if i == 0:
            yn = conv_bn(x, wA["W1_0"], wA["bn1_0"])
            print("check A torch vs w_A.npz: max abs diff", float((ya - yn).abs().max()), flush=True)
            assert float((ya - yn).abs().max()) < 1e-3
            x2 = x[:1].clone(); x2[0, :, 6, 8] += 5.0
            d = (a0.bn1(a0.conv1(x2)) - ya[:1]).abs().sum(1)[0]
            moved = torch.nonzero(d > 1e-4).cpu().tolist()
            assert moved == [[3, 4]], moved
            print("check mapping: stage-3 (6, 8) moves only cell (3, 4): PASS", flush=True)
        yb = conv_bn(x, wB["W1_0"], wB["bn1_0"])
        pw = F.max_pool2d(x[:, 69:70], 3, stride=2, padding=1)[:, 0]
        Ap.append(ya.flatten(2).half().cpu()); Bp.append(yb.flatten(2).half().cpu()); PW.append(pw.flatten(1).cpu())
    Ap, Bp, PW = torch.cat(Ap).numpy(), torch.cat(Bp).numpy(), torch.cat(PW).numpy()
    assert Bp.shape[1] == 128, Bp.shape
    out = os.path.join(HERE, "..", "results", "oc_tree" + ("_smoke" if smoke else "") + ".npz")
    np.savez(out, ids=ids, A_pre=Ap, B_pre=Bp, pwin=PW)
    print("saved", out, Ap.shape, Bp.shape, PW.shape)


if __name__ == "__main__":
    main()
