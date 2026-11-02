# serves: H-scale
"""Part C, Hypothesis H-scale: does a fixed map of the last hidden layer get cleaner as the network grows?
torchvision ResNet-18, 34, 50, 101, 152 (one recipe). Mix: the 80 dog photos and 80 car photos of `pc_map2.py`
(same draw, seed 29, dev split), weight 1/2 per region.

Map: regions "dog photos" and "car photos"; every channel of layer4 (pooled over the 7 x 7 positions, which is exact for
what the head reads, since the head is linear in the pooled values) goes to the region that holds most of its activity
(so the map covers the layer). leak_c(x) = h_c(x) at the photos outside c's region.
Measured per net:
  Lambda      sum_x p_x osc(W leak(x)) / sum_x p_x osc(y_N(x) - b)   (H-scale's output-unit district error; osc over
              all 1000 logits, as in Lemma TE0)
  KL_leak     mean KL(N || N with every leak set to 0, head kept)    (what N really sends through the leaks)
  agree_leak  share of photos whose top class is unchanged
  share_med   median over channels of the DM2 share (activity outside the channel's region / total)
  share_act   activity-weighted mean share (each channel weighted by its total activity)
  share_out   output-weighted mean share (each channel weighted by activity x |its dog-minus-car logit weight|)
  top writers the 10 channels with the largest activity x |dog-minus-car weight|, their shares (check on the units that
              matter, lesson of 2026-10-28)
Spread: 200 bootstrap resamples of the photos (within each region) for Lambda, KL_leak and share_out.
Usage: python pc_scale.py [n_per_region]   (smoke: 16; full: 80)
"""
import json
import os
import sys

import numpy as np
import torch

import resnets as R

HERE = os.path.dirname(os.path.abspath(__file__))
n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 80
rng = np.random.default_rng(29)
CARS = [407, 436, 468, 511, 555, 569, 609, 627, 656, 661, 675, 717, 734, 751, 817, 864]
X, y = R.load_images()
_, dev, _ = R.split(y)
dog = dev[(y[dev] >= 151) & (y[dev] <= 268)]
car = dev[np.isin(y[dev], CARS)]
ix = np.r_[rng.choice(dog, n_per, replace=False), rng.choice(car, min(n_per, len(car)), replace=False)]
isdog = np.r_[np.ones(n_per, bool), np.zeros(min(n_per, len(car)), bool)]
DOGC = np.arange(151, 269)
brng = np.random.default_rng(7)


def kl(Y0, Y1):
    l0 = Y0 - np.log(np.exp(Y0 - Y0.max(1, keepdims=True)).sum(1, keepdims=True)) - Y0.max(1, keepdims=True)
    l1 = Y1 - np.log(np.exp(Y1 - Y1.max(1, keepdims=True)).sum(1, keepdims=True)) - Y1.max(1, keepdims=True)
    return (np.exp(l0) * (l0 - l1)).sum(1)


def measures(h, Y, W, b, sel):
    """h: pooled activities (photos x C), Y: logits, W: (1000 x C), b: bias; sel: photo indices (a resample)."""
    h = h[sel]; Y = Y[sel]; d = isdog[sel]
    p = np.where(d, 0.5 / d.sum(), 0.5 / (~d).sum())
    A_dog = (p[d, None] * h[d]).sum(0); A_car = (p[~d, None] * h[~d]).sum(0)
    reg_dog = A_dog >= A_car
    leak = np.where(d[:, None] != reg_dog[None, :], h, 0.0)          # activity outside the channel's region
    WL = leak @ W.T
    Lam = float(p @ (WL.max(1) - WL.min(1))) / float(p @ ((Y - b).max(1) - (Y - b).min(1)))
    Yc = Y - WL
    KLl = float(p @ kl(Y, Yc))
    agree = float(p @ (Yc.argmax(1) == Y.argmax(1)))
    tot = A_dog + A_car
    share = np.where(tot > 0, np.minimum(A_dog, A_car) / np.maximum(tot, 1e-30), 0.0)
    wgap = np.abs(W[DOGC].mean(0) - W[CARS].mean(0))
    wout = tot * wgap
    return dict(Lambda=Lam, KL_leak=KLl, agree_leak=agree, share_med=float(np.median(share[tot > 0])),
                share_act=float((tot * share).sum() / tot.sum()), share_out=float((wout * share).sum() / wout.sum()),
                top_writers=[(int(c), float(share[c])) for c in np.argsort(-wout)[:10]])


out = dict(n_per=n_per, nets={})
for name in R.SIZES:
    m = R.load(name)
    H, Y = R.acts(m, X, ix)
    h = H.mean(axis=(2, 3))
    W = m.fc.weight.detach().cpu().numpy(); b = m.fc.bias.detach().cpu().numpy()
    full = measures(h, Y, W, b, np.arange(len(ix)))
    boots = []
    di, ci = np.where(isdog)[0], np.where(~isdog)[0]
    for _ in range(200):
        sel = np.r_[brng.choice(di, len(di)), brng.choice(ci, len(ci))]
        r = measures(h, Y, W, b, sel)
        boots.append([r["Lambda"], r["KL_leak"], r["share_out"]])
    se = np.array(boots).std(0)
    acc = float((Y.argmax(1) == y[ix]).mean())
    full.update(params=R.n_params(m), C=int(h.shape[1]), top1=acc, se_Lambda=float(se[0]), se_KL_leak=float(se[1]),
                se_share_out=float(se[2]))
    out["nets"][name] = full
    print(f"{name}: {R.n_params(m) / 1e6:.1f}M C={h.shape[1]} top-1 {acc:.3f} | Lambda {full['Lambda']:.4f} "
          f"(se {se[0]:.4f}) | KL_leak {full['KL_leak']:.4f} (se {se[1]:.4f}) agree {full['agree_leak']:.3f} | "
          f"share med {full['share_med']:.3f} act {full['share_act']:.3f} out {full['share_out']:.3f} (se {se[2]:.3f})",
          flush=True)
    print("   top writers (channel, share):", ", ".join(f"{c}:{s:.3f}" for c, s in full["top_writers"]))
    del m
    if R.DEV.type == "mps":
        torch.mps.empty_cache()
json.dump(out, open(os.path.join(HERE, "results_pc_scale.json"), "w"), indent=1)
