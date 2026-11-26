"""Step 4c of "Children by the parent's write" (LOG, 2026-11-23): confirm the parent fold on fresh photos (train cache).
Parent fold: read = a-weighted mean of the k children's channel-69 kernels only, bias = a-weighted mean of their bn1
shifts, write = sum of their writes. Controls: delete, full fold (step 4), n297's true output through all k writes.
Runs on the cluster in n6/clean/neuron/coarse. Saved: results/coarse/r18/oc_nuudle3.npz.
Usage: python -u oc_nuudle3.py [n]   (a number = smoke run on the first n parent-strong and n random photos)
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
P, KS, ONE = 69, (6, 11, 20, 40), 297


def bn_parts(bn):
    sc = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return sc, bn.bias - bn.running_mean * sc


def main():
    smoke = int(sys.argv[1]) if len(sys.argv) > 1 else None
    m = net_for("A"); b0, b1 = m.layer4[0], m.layer4[1]
    s1, t1 = bn_parts(b0.bn1)
    s2, _ = bn_parts(b0.bn2)
    R = b0.conv1.weight * s1.view(-1, 1, 1, 1)
    wnorm = torch.sqrt(((b0.conv2.weight * s2.view(-1, 1, 1, 1)) ** 2).sum((0, 2, 3)))
    score = (R[:, P].sum((1, 2)) * wnorm).numpy()
    order = np.argsort(-score)
    print("top 20:", order[:20].tolist(), flush=True)
    tr = r18.Cache("train")
    peak = np.concatenate([torch.from_numpy(np.ascontiguousarray(tr.x[s:s + 1000, P])).float().amax((1, 2)).numpy()
                           for s in range(0, len(tr), 1000)])
    strong_rows = np.nonzero(peak >= 2.92)[0]
    rand_rows = np.sort(np.random.default_rng(2).choice(len(tr), 2000, replace=False))
    if smoke:
        strong_rows, rand_rows = strong_rows[:smoke], rand_rows[:smoke]
    print(f"train photos {len(tr)}, parent-strong {len(strong_rows)}, random {len(rand_rows)}", flush=True)

    def fold_out(x, K, a, only69):
        w = torch.as_tensor(a / a.sum(), dtype=torch.float32)
        rf = (R[K] * w.view(-1, 1, 1, 1)).sum(0, keepdim=True)
        if only69:
            keep = torch.zeros_like(rf); keep[:, P] = rf[:, P]; rf = keep
        return torch.relu(F.conv2d(x, rf, stride=b0.conv1.stride, padding=b0.conv1.padding) + (t1[K] * w).sum())

    def tail(x, inner):
        h1 = torch.relu(b0.bn2(b0.conv2(inner)) + b0.downsample(x))
        return r18.net.head(m, r18.net.block_parts(b1, h1)["h"])

    res = {}
    t0 = time.time()
    for tag, rows in (("strong", strong_rows), ("random", rand_rows)):
        kl = {}
        for s in range(0, len(rows), 200):
            x, z = tr.get(rows[s:s + 200])
            inner = torch.relu(b0.bn1(b0.conv1(x)))
            kl.setdefault("base", []).append(r18.kl_rows(z, tail(x, inner)).numpy())
            for k in KS:
                K = torch.as_tensor(order[:k].copy()); a = score[order[:k]]
                for name, val in (("delete", 0), ("fold", fold_out(x, K, a, False)),
                                  ("parent", fold_out(x, K, a, True)), ("one", inner[:, ONE].unsqueeze(1))):
                    inn = inner.clone(); inn[:, K] = val
                    kl.setdefault(f"{name}_{k}", []).append(r18.kl_rows(z, tail(x, inn)).numpy())
            print(f"  {tag} {s + len(x)}/{len(rows)} {time.time() - t0:.0f} s", flush=True)
        kl = {n: np.concatenate(v) for n, v in kl.items()}
        res.update({f"{tag}_{n}": v for n, v in kl.items()})
        n = len(kl["base"])
        B = [np.random.default_rng(i).integers(0, n, n) for i in range(2000)]
        print(f"[{tag}] photos {n}, base KL max {kl['base'].max():.2e}")
        for k in KS:
            def rec(c, b):
                return 1 - kl[f"{c}_{k}"][b].mean() / kl[f"delete_{k}"][b].mean()
            bp = np.array([rec("parent", b) for b in B])
            df = np.array([rec("parent", b) - rec("fold", b) for b in B])
            do = np.array([rec("parent", b) - rec("one", b) for b in B])
            allb = np.arange(n)
            print(f"  k={k:2d} delete {kl[f'delete_{k}'].mean():.4f} | parent {rec('parent', allb):.3f} "
                  f"CI [{np.percentile(bp, 2.5):.3f}, {np.percentile(bp, 97.5):.3f}] | fold {rec('fold', allb):.3f} "
                  f"| one n{ONE} {rec('one', allb):.3f} | parent-fold CI [{np.percentile(df, 2.5):.3f}, "
                  f"{np.percentile(df, 97.5):.3f}] | parent-one CI [{np.percentile(do, 2.5):.3f}, {np.percentile(do, 97.5):.3f}]")
    out = os.path.join(r18.OUT, "oc_nuudle3" + ("_smoke" if smoke else "") + ".npz")
    np.savez(out, order=order, score=score, strong_rows=strong_rows, rand_rows=rand_rows, **res)
    print("saved", out)


if __name__ == "__main__":
    main()
