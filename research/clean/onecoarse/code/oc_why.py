# scale: neurons=s3:69, A:children; net=A; photos=train-cache, held, test; where=cluster
# rung: Q=why the trained write leaves the children's paths: no-training regression rule vs trained; N=1 unit (the parent) + 11 children; EST=10 min CPU; PREV=results/oc_morph2.npz; PERMIT=Shonu 2026-11-24 "figure out then why the trained neuron is taking an unoptimal path"
"""Why does the trained write leave the children's paths? (LOG section of the same name.)

Rule LS (no training): W_LS = argmin over train + trainflip photos and all 7 x 7 cells of |y - W window(p)|^2,
  y = the 11 children's write into block 0's residual (conv2 with bn2 scale, before the add), 512 per cell,
  window(p) = the parent's 7 x 7 window (stride 2, padding 3), 49 values. No constant. Closed form.
  Derived from training's fixed point E[H (W window(p) - y) window(p)] = 0 with H equal in every direction.
Builds (children deleted in all): calc (factored, as oc_morph.py), trained s0 and s1 (oc_morph2.npz), ls.
Reports recovery on test parent-strong and test random (oc_morph rows). On held (DEV): the exact logit split
T = base - delete, Ro = base - routecut, O = routecut - delete, and each build's restoration M = build - delete.
Saved: results/coarse/r18/oc_why.npz. Usage: python -u oc_why.py [smoke]
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import r18
from capture18 import net_for

torch.set_grad_enabled(False)
P, K_N = 69, 11
SMOKE = "smoke" in sys.argv
DEV = r18.DEV


def bn_parts(bn):
    sc = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return sc, bn.bias - bn.running_mean * sc


def main():
    t0 = time.time()
    m = net_for("A").to(DEV).eval()
    b0, b1 = m.layer4[0], m.layer4[1]
    s1, _ = bn_parts(b0.bn1)
    s2, _ = bn_parts(b0.bn2)
    R = b0.conv1.weight * s1.view(-1, 1, 1, 1)
    V = b0.conv2.weight * s2.view(-1, 1, 1, 1)
    wnorm = torch.sqrt((V ** 2).sum((0, 2, 3)))
    order = np.argsort(-(R[:, P].sum((1, 2)) * wnorm).cpu().numpy())
    kt = torch.as_tensor(order[:K_N].copy(), device=DEV)
    k_cal, v_cal = R[kt][:, P:P + 1].contiguous(), V[:, kt].contiguous()
    mo = np.load(os.path.join(r18.OUT, "oc_morph2.npz"))
    assert np.array_equal(mo["order"][:K_N], order[:K_N])
    trs = {s: (torch.as_tensor(mo[f"tr{s}_k"], device=DEV), torch.as_tensor(mo[f"tr{s}_v"], device=DEV)) for s in (0, 1)}

    tr, trf, held, test = (r18.Cache(n) for n in ("train", "trainflip", "held", "test"))
    print(f"train {len(tr)}, trainflip {len(trf)}, held {len(held)}, test {len(test)}; dev {DEV}", flush=True)

    def kids_write(x):
        h = torch.relu(b0.bn1(b0.conv1(x)))[:, kt]
        return F.conv2d(h, v_cal, padding=1)                                       # B x 512 x 7 x 7

    def window(x):
        return F.unfold(x[:, P:P + 1], 7, padding=3, stride=2)                     # B x 49 x 49 (features x cells)

    # normal equations
    XtX = torch.zeros(49, 49, dtype=torch.float64, device=DEV)
    XtY = torch.zeros(49, 512, dtype=torch.float64, device=DEV)
    for c in (tr, trf):
        rows = np.arange(len(c))[:(500 if SMOKE else None)]
        for s in range(0, len(rows), 500):
            x, _ = c.get(rows[s:s + 500])
            X = window(x).double()
            Y = kids_write(x).flatten(2).double()                                  # B x 512 x 49
            XtX += torch.einsum("bfl,bgl->fg", X, X)
            XtY += torch.einsum("bfl,bol->fo", X, Y)
    W = torch.linalg.solve(XtX + 1e-6 * torch.eye(49, device=DEV, dtype=torch.float64), XtY)   # 49 x 512
    W_ls = W.T.reshape(512, 1, 7, 7).float().contiguous()
    print(f"LS solved; {time.time()-t0:.0f} s", flush=True)

    def run(x, mode):
        pre = b0.bn1(b0.conv1(x))
        inner = torch.relu(pre)
        p = x[:, P:P + 1]
        if mode != "base":
            inner = inner.clone()
            inner[:, kt] = torch.relu(pre[:, kt] - F.conv2d(p, k_cal, stride=2, padding=1)) if mode == "routecut" else 0
        r = b0.bn2(b0.conv2(inner))
        if mode == "calc":
            r = r + F.conv2d(F.conv2d(p, k_cal, stride=2, padding=1), v_cal, padding=1)
        elif mode in ("tr0", "tr1"):
            kk, vv = trs[int(mode[-1])]
            r = r + F.conv2d(F.conv2d(p, kk, stride=2, padding=1), vv, padding=1)
        elif mode == "ls":
            r = r + F.conv2d(p, W_ls, stride=2, padding=3)
        h1 = torch.relu(r + b0.downsample(x))
        return r18.net.head(m, r18.net.block_parts(b1, h1)["h"])

    builds = ["delete", "calc", "tr0", "tr1", "ls"]
    res = dict(order=order, W_ls=W_ls.cpu().numpy(), XtX=XtX.cpu().numpy())

    def peaks(c, rows):
        return np.concatenate([c.get(rows[s:s + 500])[0][:, P].amax((1, 2)).cpu().numpy() for s in range(0, len(rows), 500)])
    sets = {"test_strong": mo["test_strong_rows"], "test_random": mo["test_random_rows"]}
    for name, rows in sets.items():
        rows = rows[:(40 if SMOKE else None)]
        kl = {b: [] for b in builds}
        for s in range(0, len(rows), 250):
            x, z = test.get(rows[s:s + 250])
            for b in builds:
                kl[b].append(r18.kl_rows(z, run(x, b)).cpu().numpy())
        kl = {b: np.concatenate(v) for b, v in kl.items()}
        line = [f"[{name}] n {len(rows)}"] + [f"{b} {1 - kl[b].mean() / kl['delete'].mean():.3f}" for b in builds[1:]]
        print(" | ".join(line), flush=True)
        res.update({f"{name}_{b}": v for b, v in kl.items()})
        res[f"{name}_rows"] = rows

    # exact logit split on held (DEV)
    hrows = np.arange(len(held))[:(40 if SMOKE else None)]
    res["dev_peak"] = peaks(held, hrows)
    acc = {k: [] for k in ["TT", "RoRo", "OO", "RoT"] + [f"{b}_{q}" for b in ("calc", "tr0", "tr1", "ls") for q in ("MT", "MRo", "MO", "MM")]}
    for s in range(0, len(hrows), 250):
        x, _ = held.get(hrows[s:s + 250])
        cen = lambda y: y - y.mean(1, keepdim=True)
        yb, yd, yr = cen(run(x, "base")), cen(run(x, "delete")), cen(run(x, "routecut"))
        T, Ro, O = yb - yd, yb - yr, yr - yd
        dot = lambda a, b: (a * b).sum(1).cpu().numpy()
        acc["TT"].append(dot(T, T)); acc["RoRo"].append(dot(Ro, Ro)); acc["OO"].append(dot(O, O)); acc["RoT"].append(dot(Ro, T))
        for b in ("calc", "tr0", "tr1", "ls"):
            M = cen(run(x, b)) - yd
            acc[f"{b}_MT"].append(dot(M, T)); acc[f"{b}_MRo"].append(dot(M, Ro))
            acc[f"{b}_MO"].append(dot(M, O)); acc[f"{b}_MM"].append(dot(M, M))
    res.update({f"dev_{k}": np.concatenate(v) for k, v in acc.items()})
    out = os.path.join(r18.OUT, "oc_why" + ("_smoke" if SMOKE else "") + ".npz")
    np.savez(out, **res)
    print("saved", out, f"{time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
