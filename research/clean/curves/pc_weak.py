# serves: H-weak
"""Part C, Hypothesis H-weak: do a network's own readers send little through the weak activations of its last hidden
layer? torchvision ResNet-18 to 152 (one recipe, five sizes), one dev photo per ImageNet class (1000 photos).

For each net and each threshold theta: set to 0 every last-layer entry (channel x position) below theta times its
channel's largest value over the mix, keep N's head, and measure
  KL      mean KL(softmax y_N || softmax y_theta)   (the change of the output)
  agree   share of photos whose top class is unchanged
  mass    share of the layer's total activity that was removed
  entries share of the positive entries that were removed
  bound   mean osc(y_theta - y_N) (Lemma TE0 bounds KL by it; the term (d) quantity of Theorem TE4)
Plus the essay-style cut: keep the strongest quarter of each photo's positive entries.
Check on known units first (lesson of 2026-10-28): top-1 accuracy of each net on these photos against its published
figure, so the preprocessing is right before any number is read.
Usage: python pc_weak.py [n_photos]   (smoke: 100; full: 1000)
"""
import json
import os
import sys

import numpy as np
import torch

import resnets as R

HERE = os.path.dirname(os.path.abspath(__file__))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
rng = np.random.default_rng(31)
X, y = R.load_images()
_, dev, _ = R.split(y)
per_class = {}
for i in dev:
    per_class.setdefault(int(y[i]), []).append(int(i))
ix = np.array(sorted(rng.choice(per_class[c]) for c in sorted(per_class)))
ix = np.sort(rng.choice(ix, n, replace=False)) if n < len(ix) else ix
PUBLISHED = dict(resnet18=69.76, resnet34=73.31, resnet50=76.13, resnet101=77.37, resnet152=78.31)
THETAS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]


def kl(Y0, Y1):
    l0 = torch.log_softmax(Y0, -1); l1 = torch.log_softmax(Y1, -1)
    return (l0.exp() * (l0 - l1)).sum(-1)


out = dict(n=int(len(ix)), nets={})
for name in R.SIZES:
    m = R.load(name)
    H, Y = R.acts(m, X, ix)
    acc = float((Y.argmax(1) == y[ix]).mean() * 100)
    Ht = torch.from_numpy(H); Yt = torch.from_numpy(Y)
    cmax = Ht.amax(dim=(0, 2, 3), keepdim=True).clamp_min(1e-12)
    pos = Ht > 0
    total = float(Ht.sum())
    rows = []
    for th in THETAS:
        keep = Ht >= th * cmax
        Hk = Ht * keep
        with torch.no_grad():
            Yk = R.head(m, Hk.to(R.DEV)).cpu()
        d = Yk - Yt
        rows.append(dict(theta=th, KL=float(kl(Yt, Yk).mean()), agree=float((Yk.argmax(1) == Yt.argmax(1)).float().mean()),
                         mass=float((Ht * ~keep).sum() / total), entries=float((pos & ~keep).sum() / pos.sum()),
                         bound=float((d.amax(1) - d.amin(1)).mean())))
    # essay-style: keep the strongest quarter of each photo's positive entries
    flat = Ht.reshape(len(ix), -1)
    Hq = torch.zeros_like(flat)
    for i in range(len(ix)):
        v = flat[i]; k = int((v > 0).sum())
        if k:
            top = torch.topk(v, max(1, k // 4)).indices
            Hq[i, top] = v[top]
    with torch.no_grad():
        Yq = R.head(m, Hq.reshape(Ht.shape).to(R.DEV)).cpu()
    dq = Yq - Yt
    quarter = dict(KL=float(kl(Yt, Yq).mean()), agree=float((Yq.argmax(1) == Yt.argmax(1)).float().mean()),
                   mass=float(1 - Hq.sum() / total), acc=float((Yq.argmax(1).numpy() == y[ix]).mean() * 100),
                   bound=float((dq.amax(1) - dq.amin(1)).mean()))
    out["nets"][name] = dict(params=R.n_params(m), top1=acc, top1_published=PUBLISHED[name], C=int(H.shape[1]),
                             zero_share=float((H == 0).mean()), thresholds=rows, quarter=quarter)
    print(f"{name}: params {R.n_params(m) / 1e6:.1f}M  top-1 {acc:.1f} (published {PUBLISHED[name]})  zeros {(H == 0).mean():.3f}")
    for r in rows:
        print(f"   theta {r['theta']:.2f}: KL {r['KL']:.4f}  agree {r['agree']:.3f}  mass removed {r['mass']:.3f}"
              f"  entries removed {r['entries']:.3f}  TE0 bound {r['bound']:.3f}")
    print(f"   quarter kept: KL {quarter['KL']:.4f} agree {quarter['agree']:.3f} mass removed {quarter['mass']:.3f}"
          f" top-1 {quarter['acc']:.1f} vs {acc:.1f}")
    del m
    if R.DEV.type == "mps":
        torch.mps.empty_cache()
json.dump(out, open(os.path.join(HERE, "results_pc_weak.json"), "w"), indent=1)
