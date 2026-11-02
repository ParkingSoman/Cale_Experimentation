# serves: H-mass
"""Part C, Hypothesis H-mass (and the two forms of H-scale): Definition DN2's numbers on torchvision ResNet-18 to 152.
Same photos, mix and map as `pc_scale.py` (80 dog and 80 car photos, weight 1/2 per region; regions dog / car; every
layer4 channel, pooled over positions, goes to the region that holds most of its activity; leaks = the activity outside
it, clean value 0). osc is over all 1000 logits.
Per net:
  O      output spread  sum_x p_x osc(W h(x))                       (H-mass part 1: does it grow?)
  M      write mass     sum_c osc(W_c) A_c
  rho    M / O          (cancellation ratio, >= 1)
  phi    fair deviation sum_c osc(W_c) leakmass_c / M               (H-scale, fair form)
  D_M    deviation sent sum_x p_x osc(W leak(x));  Lambda = D_M / O  (H-scale, quiet form; equals pc_scale's Lambda)
  rho_leak  sum_c osc(W_c) leakmass_c / D_M                         (how much the leaks cancel at the logits)
Theorem TN3 part 3 says Lambda <= rho phi; here Lambda = phi rho / rho_leak.
Spread: 200 bootstrap resamples within each region.
Usage: python pc_mass.py [n_per_region]   (full: 80)
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
brng = np.random.default_rng(7)
KEYS = ["O", "M", "rho", "phi", "D_M", "Lambda", "rho_leak"]


def osc_rows(A):
    return A.max(1) - A.min(1)


def measures(h, W, sel):
    h = h[sel]; d = isdog[sel]
    p = np.where(d, 0.5 / d.sum(), 0.5 / (~d).sum())
    A_dog = (p[d, None] * h[d]).sum(0); A_car = (p[~d, None] * h[~d]).sum(0)
    reg_dog = A_dog >= A_car
    leak = np.where(d[:, None] != reg_dog[None, :], h, 0.0)
    oscW = W.max(0) - W.min(0)
    A = A_dog + A_car
    lm = p @ leak
    O = float(p @ osc_rows(h @ W.T)); M = float(oscW @ A)
    D = float(p @ osc_rows(leak @ W.T)); num = float(oscW @ lm)
    return dict(O=O, M=M, rho=M / O, phi=num / M, D_M=D, Lambda=D / O, rho_leak=num / D)


out = dict(n_per=n_per, nets={})
for name in R.SIZES:
    m = R.load(name)
    H, Y = R.acts(m, X, ix)
    h = H.mean(axis=(2, 3))
    W = m.fc.weight.detach().cpu().numpy()
    full = measures(h, W, np.arange(len(ix)))
    di, ci = np.where(isdog)[0], np.where(~isdog)[0]
    boots = np.array([[measures(h, W, np.r_[brng.choice(di, len(di)), brng.choice(ci, len(ci))])[k] for k in KEYS]
                      for _ in range(200)])
    full.update({f"se_{k}": float(s) for k, s in zip(KEYS, boots.std(0))})
    full.update(params=R.n_params(m), C=int(h.shape[1]))
    assert full["Lambda"] <= full["rho"] * full["phi"] + 1e-9 and full["rho"] >= 1 - 1e-9   # TN3 part 3
    out["nets"][name] = full
    print(f"{name}: {R.n_params(m) / 1e6:.1f}M C={h.shape[1]} | " +
          " ".join(f"{k} {full[k]:.4f} ({full['se_' + k]:.4f})" for k in KEYS), flush=True)
    del m
    if R.DEV.type == "mps":
        torch.mps.empty_cache()
json.dump(out, open(os.path.join(HERE, "results_pc_mass.json"), "w"), indent=1)
