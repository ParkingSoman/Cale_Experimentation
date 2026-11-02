"""C5 (microscope, uses the FITTED Fisher write as a fixed instrument). Same write procedure (Fisher solve, 1500 train
images, R=10) on three reads; dev-5000 real KL, paired.
  der2 = old FITTED read (least squares, 600 images)            [reproduces W2: .241]
  c1dial = weights-only moved sum (C1 shifts), bias so it never clips (a linear unit: the bias is then a gauge)
  c1cons = weights-only moved sum, bias = sum of member biases (consensus, fires ~1%)
Prediction (written first): c1dial within .02 of der2 (paired); c1cons worse than c1dial by > .05."""
import numpy as np, torch
from inception import *
from family import Data, Family
from w_fisher import load_build, solve, real_kl
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"])
for p in m.parameters(): p.requires_grad_(False)
W5 = m.mixed3b_5x5_pre_relu_conv.weight.detach()[[i - 320 for i in fam]]; b5 = m.mixed3b_5x5_pre_relu_conv.bias.detach()[[i - 320 for i in fam]]
s = np.load("cache/c1_shifts.npz")["s"]; s = np.round(s - np.round(s.mean(0))).astype(int); ks = 5 + 2 * int(np.abs(s).max()); c = ks // 2
K = torch.zeros(32, ks, ks, device=DEV)
for i, (dy, dx) in enumerate(s): K[:, c + dy - 2:c + dy + 3, c + dx - 2:c + dx + 3] += W5[i]
# never-clip bias from weights only: pre >= -sum|negative weights| * max input; use the bound with max input over a batch? no:
# a linear unit's bias is a gauge (the write's intercept absorbs it), so any bias that never clips gives the same build.
# Bound from weights: inputs are post-ReLU bottleneck values <= Xmax; Xmax itself needs data, so check clipping empirically.
builds = {}
fd = load_build(m, fam, "derived2-r1-ks7"); builds["der2"] = fd
for tag, bias in [("c1cons", float(b5.sum())), ("c1dial", 2.0e4)]:
    fm = Family(m, fam, 1, ks=ks).to(DEV); _, _, c0 = fm.weights()
    fm.set_weights(K[None], torch.tensor([bias], device=DEV), [x.to(DEV) for x in c0]); builds[tag] = fm
with torch.no_grad():
    ins = inputs_3b(m, to_3a(m, prep(D.X[np.sort(D.tr[:100])])))
    for k, fm in builds.items(): print(k, "on-rate %.4f" % float((fm.pre(ins) > 0).float().mean()), flush=True)
tr = np.sort(np.random.default_rng(11).choice(D.tr, 1500, replace=False)); ev = {}
for k, fm in builds.items():
    cols, cbs, cov = solve(m, fam, D, fm, tr, 10, 1); ev[k] = (fm, cols, cbs); print("solved", k, flush=True)
Kl = real_kl(m, fam, D, D.dv, ev)
for k, v in Kl.items(): print(f"{k:8s} dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})")
for a, b in [("c1dial", "der2"), ("c1cons", "c1dial")]:
    d = Kl[a] - Kl[b]; print(f"paired {a} - {b}: {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})")
np.savez("cache/c5_readsplit.npz", **Kl)
