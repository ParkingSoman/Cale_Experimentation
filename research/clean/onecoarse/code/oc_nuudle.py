"""Step 4 of "Children by the parent's write" (LOG, 2026-11-23): the nuudle test in A.
Runs on the cluster in n6/clean/neuron/coarse (r18.py, capture18.py, the held cache).

Children = top k of the child score: (bn1-scaled conv1 weight on channel 69, summed over 3 x 3) x write norm
(bn2-scaled conv2 column). Fold: one neuron, read = a-weighted mean of the children's bn1-scaled reads, bias =
a-weighted mean of their bn1 shifts; write = sum of their writes (each child's output is replaced by the folded output).
Conditions: base, delete K, fold K, shared oracle (a-weighted mean of the children's true outputs), 20 random folds.
Loss = KL(cached A logits, changed logits). Reported on all held photos and on parent-strong photos.
Saved: results/coarse/r18/oc_nuudle.npz.  Usage: python -u oc_nuudle.py [n_photos]  (a number = smoke run)
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
P, KS, NRAND = 69, (6, 11, 20), 20


def bn_parts(bn):
    sc = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return sc, bn.bias - bn.running_mean * sc


def main():
    smoke = int(sys.argv[1]) if len(sys.argv) > 1 else None
    m = net_for("A"); b0, b1 = m.layer4[0], m.layer4[1]
    s1, t1 = bn_parts(b0.bn1)
    s2, _ = bn_parts(b0.bn2)
    R = b0.conv1.weight * s1.view(-1, 1, 1, 1)                     # 512 x 256 x 3 x 3
    wnorm = torch.sqrt(((b0.conv2.weight * s2.view(-1, 1, 1, 1)) ** 2).sum((0, 2, 3)))
    score = (R[:, P].sum((1, 2)) * wnorm).numpy()
    order = np.argsort(-score)
    print("top 20 by child score:", order[:20].tolist(), flush=True)

    held, test = r18.Cache("held"), r18.Cache("test")
    rows = (np.sort(np.random.default_rng(1).choice(len(held), min(2000, len(held)), replace=False))
            if smoke is None else np.arange(smoke))
    # parent-strong test photos are added (deviation logged before the full run: the smoke had 0 strong held photos)
    tstrong = np.array([i for s in range(0, len(test), 500)
                        for i in s + np.nonzero(np.asarray(test.x[s:s + 500, P]).max((1, 2)) >= 2.92)[0]], dtype=int)
    if smoke is not None:
        tstrong = tstrong[:20]
    jobs = [(held, r) for r in [rows]] + [(test, tstrong)]
    print(f"held photos {len(rows)}, parent-strong test photos {len(tstrong)}", flush=True)
    rng = np.random.default_rng(0)
    rand_sets = {k: [(rng.choice(512, k, replace=False), rng.permutation(score[order[:k]])) for _ in range(NRAND)]
                 for k in KS}

    def fold_out(x, K, a):
        a = torch.as_tensor(a, dtype=torch.float32)
        w = a / a.sum()
        rf = (R[K] * w.view(-1, 1, 1, 1)).sum(0, keepdim=True)
        bf = (t1[K] * w).sum()
        return torch.relu(F.conv2d(x, rf, stride=b0.conv1.stride, padding=b0.conv1.padding) + bf)

    def tail(x, inner):
        h1 = torch.relu(b0.bn2(b0.conv2(inner)) + b0.downsample(x))
        return r18.net.head(m, r18.net.block_parts(b1, h1)["h"])

    names = ["base"] + [f"{c}_{k}" for k in KS for c in ("delete", "fold", "oracle")] + \
            [f"rand{c}_{k}_{i}" for k in KS for i in range(NRAND) for c in ("del", "fold")]
    kl = {n: [] for n in names}
    strong, src = [], []
    t0 = time.time()
    for cache, rr_ in jobs:
        for s in range(0, len(rr_), 250):
            x, z = cache.get(rr_[s:s + 250])
            src.append(np.full(len(x), 0 if cache is held else 1))
            strong.append((x[:, P].amax((1, 2)) >= 2.92).numpy())
            inner = torch.relu(b0.bn1(b0.conv1(x)))
            kl["base"].append(r18.kl_rows(z, tail(x, inner)).numpy())
            for k in KS:
                K = torch.as_tensor(order[:k].copy()); a = score[order[:k]]
                inn = inner.clone(); inn[:, K] = 0
                kl[f"delete_{k}"].append(r18.kl_rows(z, tail(x, inn)).numpy())
                inn = inner.clone(); inn[:, K] = fold_out(x, K, a)
                kl[f"fold_{k}"].append(r18.kl_rows(z, tail(x, inn)).numpy())
                w = torch.as_tensor(a / a.sum(), dtype=torch.float32).view(1, -1, 1, 1)
                inn = inner.clone(); inn[:, K] = (inner[:, K] * w).sum(1, keepdim=True)
                kl[f"oracle_{k}"].append(r18.kl_rows(z, tail(x, inn)).numpy())
                for i, (Kr, ar) in enumerate(rand_sets[k]):
                    Kr = torch.as_tensor(Kr)
                    inn = inner.clone(); inn[:, Kr] = 0
                    kl[f"randdel_{k}_{i}"].append(r18.kl_rows(z, tail(x, inn)).numpy())
                    inn = inner.clone(); inn[:, Kr] = fold_out(x, Kr, ar)
                    kl[f"randfold_{k}_{i}"].append(r18.kl_rows(z, tail(x, inn)).numpy())
            print(f"  {s + len(x)}/{len(rr_)}  {time.time() - t0:.0f} s", flush=True)
    kl = {n: np.concatenate(v) for n, v in kl.items()}
    strong = np.concatenate(strong); src = np.concatenate(src)
    print(f"held {int((src == 0).sum())}, parent-strong held+test {int(strong.sum())}; base KL max {kl['base'].max():.2e}")
    for sub, msk in (("all held", src == 0), ("parent-strong", strong)):
        print(f"[{sub}]")
        for k in KS:
            d, f, o = (kl[f"{c}_{k}"][msk].mean() for c in ("delete", "fold", "oracle"))
            rr = [1 - kl[f"randfold_{k}_{i}"][msk].mean() / max(kl[f"randdel_{k}_{i}"][msk].mean(), 1e-12)
                  for i in range(NRAND)]
            print(f"  k={k:2d} delete {d:.4f} fold {f:.4f} oracle {o:.4f} | recovery fold {1 - f / d:.3f} "
                  f"oracle {1 - o / d:.3f} | random recovery max {max(rr):.3f} median {np.median(rr):.3f}")
    out = os.path.join(r18.OUT, "oc_nuudle" + ("_smoke" if smoke else "") + ".npz")
    np.savez(out, order=order, score=score, strong=strong, src=src, rows=rows, tstrong=tstrong, **kl)
    print("saved", out)


if __name__ == "__main__":
    main()
