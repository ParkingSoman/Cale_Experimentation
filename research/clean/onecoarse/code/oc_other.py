# scale: neurons=s3:69, A:children; net=A; photos=held, test; where=cluster
# rung: Q=does coarsening s3:69 hurt other capabilities? full logits per build; N=1 (the set of 11); EST=5 min CPU; PREV=results/oc_dial.npz; PERMIT=Shonu approved plan "plan approval opens the scale guard; does coarsening hurt other capabilities?" 2026-11-24 incl. its SCALE-UP REQUEST
"""Other capabilities (LOG "Does coarsening hurt other capabilities?").

Calculated build only. The 11 R1 children of s3:69 in A (same order rule as oc_dial.py). Builds:
  base       the network
  delete     the 11 children set to 0
  morph      delete + the parent's write through the children's paths, V on (k on p)   (= oc_dial.py)
  delmean    diagnostic: the 11 children set to their DEV mean map (11 x 7 x 7) instead of 0
  morphmean  diagnostic: children at their DEV mean map, parent writes V on (k on p - DEV mean of k on p)
Saves for DEV (2,000 held) and test (the 2,000 te_rows of oc_dial.py): base logits (fp32), and for each other build
the logit change build - base (fp16; small, so fp16 keeps it precise), plus per-photo KL (fp32) for the sanity check
against oc_dial.npz. The bars are read on the Mac (oc_other_read.py).
Usage: python -u oc_other.py [smoke]
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
P, K_SET = 69, 11
SMOKE = "smoke" in sys.argv
DEV = r18.DEV
BUILDS = ["delete", "morph", "delmean", "morphmean"]


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
    score = (R[:, P].sum((1, 2)) * torch.sqrt((V ** 2).sum((0, 2, 3)))).cpu().numpy()
    order = np.argsort(-score)
    kids = order[:K_SET].copy()
    kt = torch.as_tensor(kids, device=DEV)
    kP = R[kt][:, P:P + 1]
    print("kids:", kids.tolist(), flush=True)

    held, test = r18.Cache("held"), r18.Cache("test")
    dev_rows = np.arange(len(held))
    te_rows = np.sort(np.random.default_rng(2).choice(len(test), 2000, replace=False))   # = oc_dial / oc_morph
    if SMOKE:
        dev_rows, te_rows = dev_rows[:40], te_rows[:40]

    def parts(x):
        pre = b0.bn1(b0.conv1(x))
        push = F.conv2d(x[:, P:P + 1], kP, stride=2, padding=1)                    # B x 11 x 7 x 7
        return pre, push

    # DEV means of the children's values and of the parent's push (all 2,000 held photos, even in smoke)
    sc, sp, n = 0, 0, 0
    allrows = np.arange(len(held))
    for s in range(0, len(allrows), 250):
        x, _ = held.get(allrows[s:s + 250])
        pre, push = parts(x)
        sc = sc + torch.relu(pre[:, kt]).sum(0)
        sp = sp + push.sum(0)
        n += len(x)
    cmean, pmean = (sc / n).unsqueeze(0), (sp / n).unsqueeze(0)                      # 1 x 11 x 7 x 7
    print(f"DEV means: children {float(cmean.mean()):.4f}, push {float(pmean.mean()):.4f}; {time.time()-t0:.0f} s",
          flush=True)

    def build(x, mode):
        pre, push = parts(x)
        inner = torch.relu(pre)
        if mode != "base":
            inner = inner.clone()
            inner[:, kt] = cmean.expand(len(x), -1, -1, -1) if mode in ("delmean", "morphmean") else 0
        r = b0.bn2(b0.conv2(inner))
        if mode == "morph":
            r = r + F.conv2d(push, V[:, kt], padding=1)
        elif mode == "morphmean":
            r = r + F.conv2d(push - pmean, V[:, kt], padding=1)
        h1 = torch.relu(r + b0.downsample(x))
        return r18.net.head(m, r18.net.block_parts(b1, h1)["h"])

    res = dict(order=order, kids=kids, dev_rows=dev_rows, te_rows=te_rows, cmean=cmean.cpu().numpy(),
               pmean=pmean.cpu().numpy())
    for tag, cache, rows in (("dev", held, dev_rows), ("te", test, te_rows)):
        base, dif, kl, err = [], {b: [] for b in BUILDS}, {b: [] for b in ["base"] + BUILDS}, 0.0
        peak = []
        for s in range(0, len(rows), 250):
            x, z = cache.get(rows[s:s + 250])
            yb = build(x, "base")
            err = max(err, float((yb - z).abs().max()))
            base.append(yb.float().cpu().numpy())
            peak.append(x[:, P].amax((1, 2)).float().cpu().numpy())
            kl["base"].append(r18.kl_rows(z, yb).cpu().numpy())
            for b in BUILDS:
                y = build(x, b)
                dif[b].append((y - yb).half().cpu().numpy())
                kl[b].append(r18.kl_rows(z, y).cpu().numpy())
        res[f"{tag}_base"] = np.concatenate(base)
        res[f"{tag}_peak"] = np.concatenate(peak)
        for b in BUILDS:
            res[f"{tag}_d_{b}"] = np.concatenate(dif[b])
        for b in kl:
            res[f"{tag}_kl_{b}"] = np.concatenate(kl[b])
        print(f"[{tag}] base max abs diff from cache {err:.2e}; base KL max {res[f'{tag}_kl_base'].max():.1e}; "
              + "; ".join(f"{b} KL {res[f'{tag}_kl_{b}'].mean():.5f}" for b in BUILDS) + f"; {time.time()-t0:.0f} s",
              flush=True)
        assert res[f"{tag}_kl_base"].max() < 1e-4
    out = os.path.join(r18.OUT, "oc_other" + ("_smoke" if SMOKE else "") + ".npz")
    np.savez(out, **res)
    print("saved", out, f"{time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
