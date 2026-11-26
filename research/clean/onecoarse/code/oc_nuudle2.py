"""Step 4b of "Children by the parent's write" (LOG, 2026-11-23): harder controls for the fold.
Runs on the cluster in n6/clean/neuron/coarse. Photos: the 389 parent-strong photos of oc_nuudle.npz
(held + test, s3:69 peak >= 2.92); the 2,000 held photos only for each child's mean output.
Conditions per k in (6, 11, 20): delete, fold (as step 4), one child shared (each child i's true output through all
k writes), parent only (folded read with only channel 69 kept), mean output (each child's mean per cell).
Saved: results/coarse/r18/oc_nuudle2.npz (per-photo KL per condition).  Usage: python -u oc_nuudle2.py
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import r18
from capture18 import net_for

torch.set_grad_enabled(False)
P, KS = 69, (6, 11, 20)


def bn_parts(bn):
    sc = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return sc, bn.bias - bn.running_mean * sc


def main():
    prev = np.load(os.path.join(r18.OUT, "oc_nuudle.npz"))
    m = net_for("A"); b0, b1 = m.layer4[0], m.layer4[1]
    s1, t1 = bn_parts(b0.bn1)
    s2, _ = bn_parts(b0.bn2)
    R = b0.conv1.weight * s1.view(-1, 1, 1, 1)
    wnorm = torch.sqrt(((b0.conv2.weight * s2.view(-1, 1, 1, 1)) ** 2).sum((0, 2, 3)))
    score = (R[:, P].sum((1, 2)) * wnorm).numpy()
    order = np.argsort(-score)
    assert (order[:20] == prev["order"][:20]).all()
    held, test = r18.Cache("held"), r18.Cache("test")

    mean_out = 0
    hrows = prev["rows"]
    for s in range(0, len(hrows), 250):
        x, _ = held.get(hrows[s:s + 250])
        mean_out = mean_out + torch.relu(b0.bn1(b0.conv1(x))).sum(0)
    mean_out = mean_out / len(hrows)                                   # 512 x 7 x 7

    strong, src = prev["strong"].astype(bool), prev["src"]
    hs = prev["rows"][strong[:len(hrows)]]
    ts = prev["tstrong"][strong[len(hrows):]]   # step 4's in-run mask (2 threshold photos differ from the scan)
    assert len(hs) + len(ts) == strong.sum() == 389

    def fold_out(x, K, a, only69=False):
        w = torch.as_tensor(a / a.sum(), dtype=torch.float32)
        rf = (R[K] * w.view(-1, 1, 1, 1)).sum(0, keepdim=True)
        if only69:
            keep = torch.zeros_like(rf); keep[:, P] = rf[:, P]; rf = keep
        return torch.relu(F.conv2d(x, rf, stride=b0.conv1.stride, padding=b0.conv1.padding) + (t1[K] * w).sum())

    def tail(x, inner):
        h1 = torch.relu(b0.bn2(b0.conv2(inner)) + b0.downsample(x))
        return r18.net.head(m, r18.net.block_parts(b1, h1)["h"])

    kl = {}
    def add(name, v):
        kl.setdefault(name, []).append(v.numpy())
    t0 = time.time()
    for cache, rr in ((held, hs), (test, ts)):
        for s in range(0, len(rr), 200):
            x, z = cache.get(rr[s:s + 200])
            inner = torch.relu(b0.bn1(b0.conv1(x)))
            add("base", r18.kl_rows(z, tail(x, inner)))
            for k in KS:
                K = torch.as_tensor(order[:k].copy()); a = score[order[:k]]
                inn = inner.clone(); inn[:, K] = 0; add(f"delete_{k}", r18.kl_rows(z, tail(x, inn)))
                inn = inner.clone(); inn[:, K] = fold_out(x, K, a); add(f"fold_{k}", r18.kl_rows(z, tail(x, inn)))
                inn = inner.clone(); inn[:, K] = fold_out(x, K, a, True); add(f"parent_{k}", r18.kl_rows(z, tail(x, inn)))
                inn = inner.clone(); inn[:, K] = mean_out[K].unsqueeze(0); add(f"mean_{k}", r18.kl_rows(z, tail(x, inn)))
                for j in order[:k]:
                    inn = inner.clone(); inn[:, K] = inner[:, int(j)].unsqueeze(1)
                    add(f"one_{k}_{j}", r18.kl_rows(z, tail(x, inn)))
            print(f"  {s + len(x)}/{len(rr)} {time.time() - t0:.0f} s", flush=True)
    kl = {n: np.concatenate(v) for n, v in kl.items()}
    print(f"photos {len(kl['base'])}; base KL max {kl['base'].max():.2e}")
    rng = np.random.default_rng(0)
    B = [rng.integers(0, len(kl["base"]), len(kl["base"])) for _ in range(2000)]

    def rec(c, k, b=None):
        b = slice(None) if b is None else b
        return 1 - kl[f"{c}_{k}"][b].mean() / kl[f"delete_{k}"][b].mean()

    for k in KS:
        ones = {int(j): 1 - kl[f"one_{k}_{j}"].mean() / kl[f"delete_{k}"].mean() for j in order[:k]}
        jb = max(ones, key=ones.get)
        line = f"k={k:2d} fold {rec('fold', k):.3f} parent-only {rec('parent', k):.3f} mean-output {rec('mean', k):.3f} " \
               f"best one child n{jb} {ones[jb]:.3f}"
        for ctl, key in (("parent", f"parent_{k}"), ("one", f"one_{k}_{jb}")):
            dif = [(1 - kl[f"fold_{k}"][b].mean() / kl[f"delete_{k}"][b].mean())
                   - (1 - kl[key][b].mean() / kl[f"delete_{k}"][b].mean()) for b in B]
            line += f" | fold-{ctl} CI [{np.percentile(dif, 2.5):.3f}, {np.percentile(dif, 97.5):.3f}]"
        print(line)
        print("    each child shared:", {f"n{j}": round(v, 3) for j, v in ones.items()})
    out = os.path.join(r18.OUT, "oc_nuudle2.npz")
    np.savez(out, order=order, **kl)
    print("saved", out)


if __name__ == "__main__":
    main()
