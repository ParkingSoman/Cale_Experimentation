# scale: neurons=s3:69, A:children; net=A; photos=train-cache, held; where=cluster
# rung: Q=debug: why the read-lr-0 control trains worse than oc_morph's write-only run; N=1; EST=3 min GPU; PREV=results/oc_morph2.npz
"""Debug for oc_morph_rw.py. Same batch, same (kk, vv) = oc_morph2 trained s0: loss via the cached stage-3 output
against loss via channel 69 rebuilt from t (read unchanged). Then one gradient step each from the same start:
compare the gradients on vv."""
import os

import numpy as np
import torch
import torch.nn.functional as F

import r18
import oc_morph_rw as rw
from capture18 import net_for

DEV = r18.DEV
P = 69


def main():
    m = net_for("A").to(DEV).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    b0, b1 = m.layer4[0], m.layer4[1]
    s1, _ = rw.bn_parts(b0.bn1)
    s2, _ = rw.bn_parts(b0.bn2)
    R = b0.conv1.weight * s1.view(-1, 1, 1, 1)
    V = b0.conv2.weight * s2.view(-1, 1, 1, 1)
    wnorm = torch.sqrt((V ** 2).sum((0, 2, 3)))
    order = np.argsort(-(R[:, P].sum((1, 2)) * wnorm).cpu().numpy())
    K = torch.as_tensor(order[:11].copy(), device=DEV)
    b3 = m.layer3[1]
    s3s, s3b = rw.bn_parts(b3.bn2)
    w0 = (b3.conv2.weight[P] * s3s[P]).unsqueeze(0).contiguous()
    bb = s3b[P].view(1)
    mo = np.load(os.path.join(r18.OUT, "oc_morph2.npz"))
    kk = torch.as_tensor(mo["tr0_k"], device=DEV)
    vv = torch.as_tensor(mo["tr0_v"], device=DEV)

    def run(x, t, s, rebuild, kk, vv):
        if rebuild:
            p = torch.relu(F.conv2d(t, w0, padding=1) + bb.view(1, 1, 1, 1) + s)
            x = torch.cat([x[:, :P], p, x[:, P + 1:]], 1)
        inner = torch.relu(b0.bn1(b0.conv1(x))).clone(); inner[:, K] = 0
        r = b0.bn2(b0.conv2(inner)) + F.conv2d(F.conv2d(x[:, P:P + 1], kk, stride=2, padding=1), vv, padding=1)
        h1 = torch.relu(r + b0.downsample(x))
        return r18.net.head(m, r18.net.block_parts(b1, h1)["h"])

    for gpu in (False, True):
        for name in ("train", "trainflip"):
            c = rw.TCache(name, gpu=gpu)
            i = np.random.default_rng(0).choice(len(c), 256, replace=False)
            x, z, t, s = c.get(i)
            with torch.no_grad():
                pdiff = float((torch.relu(F.conv2d(t, w0, padding=1) + bb.view(1, 1, 1, 1) + s) - x[:, P:P + 1]).abs().max())
                l0 = F.kl_div(F.log_softmax(run(x, t, s, False, kk, vv), 1), F.log_softmax(z, 1), log_target=True, reduction="batchmean")
                l1 = F.kl_div(F.log_softmax(run(x, t, s, True, kk, vv), 1), F.log_softmax(z, 1), log_target=True, reduction="batchmean")
            print(f"gpu={gpu} {name}: p rebuilt vs cache max diff {pdiff:.4f}; loss cache {float(l0):.5f}, rebuilt {float(l1):.5f}; "
                  f"x dtype {x.dtype}, t mean {float(t.mean()):.4f}, s mean {float(s.mean()):.4f}", flush=True)
            v0 = torch.zeros_like(vv, requires_grad=True)
            g = []
            for rb in (False, True):
                y = run(x, t, s, rb, kk, v0)
                loss = F.kl_div(F.log_softmax(y, 1), F.log_softmax(z, 1), log_target=True, reduction="batchmean")
                g.append(torch.autograd.grad(loss, v0)[0])
            print(f"   grad on vv from 0: cos {float((g[0] * g[1]).sum() / g[0].norm() / g[1].norm()):.4f}, "
                  f"norm ratio {float(g[1].norm() / g[0].norm()):.3f}", flush=True)


if __name__ == "__main__":
    main()
