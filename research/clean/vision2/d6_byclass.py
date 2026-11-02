"""D6 image-level test of the healing mechanism. Per sealed image: amplification 1/r = sigma_I/sigma_D (mean over
positions, block 17's LN), the volume neurons' activity (top-20 volume neurons of block 16), and the drop's KL damage.
Grouped by ImageNet range: dogs 151-268, other mammals 269-397 (+ a few), birds 7-24/80-100/127-146, reptiles/amphib
25-68, man-made 400-999. Also the correlation across images of amplification with damage."""
import os, numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); b16, b17 = m.blocks[16], m.blocks[17]; gelu = nn.gelu
r = np.load(os.path.join(D, "d6_roles.npz")); V = np.argsort(-r["vol"])[:20]; names = [n.split(",")[0] for n in class_names()]
amp, act, kl, top1 = [], [], [], []
sd = lambda v: np.sqrt(np.array(b17.conv_dw(v)).var(-1) + 1e-6)
for i in range(0, 3000, 200):
    x = mx.array(z["XE"][i:i + 200]).astype(mx.float32); x1 = b16(x)
    amp.append((sd(x1) / sd(x)).reshape(x.shape[0], -1).mean(1)); act.append(np.array(gelu(b16.fc1(b16.pre(x)))[..., mx.array(V)].mean((1, 2, 3))))
    L = m.head(b17(x)); T = mx.array(z["LE"][i:i + 200]); lp = L - mx.logsumexp(L, 1, keepdims=True); tp = T - mx.logsumexp(T, 1, keepdims=True)
    kl.append(np.array((mx.exp(tp) * (tp - lp)).sum(1))); top1.append(np.array(L.argmax(1)) == z["LE"][i:i + 200].argmax(1))
amp, act, kl, ag = (np.concatenate(v) for v in (amp, act, kl, top1)); y = z["yE"]
grp = {"dogs": (y >= 151) & (y <= 268), "other mammals": (y >= 269) & (y <= 397), "birds": ((y >= 7) & (y <= 24)) | ((y >= 80) & (y <= 100)) | ((y >= 127) & (y <= 146)),
       "reptiles+amphibians": (y >= 25) & (y <= 68), "man-made": y >= 400}
print(f"{'group':20s} {'n':>5s} {'amplification':>13s} {'volume act':>10s} {'KL damage':>9s} {'agree w/ intact':>15s}")
for g, s in grp.items(): print(f"{g:20s} {s.sum():5d} {amp[s].mean():13.3f} {act[s].mean():10.3f} {kl[s].mean():9.4f} {ag[s].mean():15.3f}")
print(f"across images: corr(amplification, volume activity) {np.corrcoef(amp, act)[0, 1]:.3f}; corr(amplification, KL) {np.corrcoef(amp, kl)[0, 1]:.3f}; corr(volume act, KL) {np.corrcoef(act, kl)[0, 1]:.3f}")
o = np.argsort(-kl)[:12]; print("12 most damaged images:", [names[y[k]] for k in o])
