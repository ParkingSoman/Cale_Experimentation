"""Step 0b (FANOUT plan): does the generic input predict ResNet-18's stored BatchNorm statistics (the known answer)?
For every BN layer: the mean and variance of its input per channel, on (a) pink noise (alpha 1), (b) white noise,
(c) 1,000 photos (reading only, as the ceiling of this check). Score = Spearman rank correlation with the stored
running_var (and running_mean), and the median ratio measured / stored variance. Output: fanout/check_bn.json.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import board  # noqa: E402
import generic  # noqa: E402


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def stats(A, x, bs=50):
    acc = {}
    hooks = []
    for name, m in A.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            def h(mod, inp, out, name=name):
                t = inp[0].float()
                s1, s2, n = acc.get(name, (0, 0, 0))
                acc[name] = (s1 + t.sum((0, 2, 3)), s2 + t.pow(2).sum((0, 2, 3)), n + t.shape[0] * t.shape[2] * t.shape[3])
            hooks.append(m.register_forward_hook(h))
    for a in range(0, len(x), bs):
        A(x[a:a + bs].to(board.DEV).float())
    for h in hooks:
        h.remove()
    out = {}
    for k, (s1, s2, n) in acc.items():
        mu = s1 / n
        out[k] = (mu.cpu().numpy(), (s2 / n - mu * mu).cpu().numpy())
    return out


def main():
    A = board.r18_A().to(board.DEV).eval()
    xs = {"pink": generic.pink(500, seed=2), "white": torch.randn(500, 3, 224, 224, generator=torch.Generator().manual_seed(2))}
    xp, _ = board.r18_x("held", 2000)
    xs["photos (reading only)"] = xp[:1000]
    bn = {n: m for n, m in A.named_modules() if isinstance(m, torch.nn.BatchNorm2d)}
    res = {}
    for key, x in xs.items():
        st = stats(A, x)
        rows = []
        for n, m in bn.items():
            mu, var = st[n]
            rv, rm = m.running_var.cpu().numpy(), m.running_mean.cpu().numpy()
            rows.append(dict(layer=n, var_rank=spearman(var, rv), mean_rank=spearman(mu, rm),
                             var_ratio_median=float(np.median(var / rv))))
        res[key] = rows
        vr = [r["var_rank"] for r in rows]
        mr = [r["mean_rank"] for r in rows]
        rat = [r["var_ratio_median"] for r in rows]
        print(f"{key}: var rank corr median {np.median(vr):.2f} (min {min(vr):.2f}), mean rank corr median {np.median(mr):.2f}, "
              f"var ratio median {np.median(rat):.2f} (range {min(rat):.2f}-{max(rat):.2f})")
    json.dump(res, open(os.path.join(HERE, "check_bn.json"), "w"), indent=1)
    for i, r in enumerate(res["pink"]):
        if i % 4 == 0:
            p = res["photos (reading only)"][i]
            print(f"  {r['layer']:28s} pink var {r['var_rank']:.2f} mean {r['mean_rank']:.2f} ratio {r['var_ratio_median']:.2f} | photos var {p['var_rank']:.2f} ratio {p['var_ratio_median']:.2f}")


if __name__ == "__main__":
    main()
