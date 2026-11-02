"""I1: calibration of InceptionV1 on the train split (cached teacher logits; images only check)."""
import numpy as np
from inception import load_images, split
from scipy.special import softmax
_, y = load_images(); tr, dv, se = split(y); T = np.load("cache/teacher_logits.npy", mmap_mode="r"); lab = np.load("label_map.npy")
P = softmax(np.asarray(T[tr], np.float64), 1); pt = P[np.arange(len(tr)), lab[y[tr]]]
Pr = P[:, lab]; conf = Pr.max(1); acc = (Pr.argmax(1) == y[tr])
print(f"n {len(tr)} | mass on the 1000 real classes {Pr.sum(1).mean():.4f} | top-1 {acc.mean():.4f} | mean confidence {conf.mean():.4f}")
print(f"E[p(true)] {pt.mean():.4f} vs E[sum p^2] {(P ** 2).sum(1).mean():.4f} (equal if calibrated) | NLL {-np.log(pt).mean():.4f}")
bins = np.linspace(0, 1, 16); ece = 0
print("bin: conf range | n | mean conf | accuracy")
for a, b in zip(bins[:-1], bins[1:]):
    s = (conf > a) & (conf <= b)
    if s.sum(): ece += s.mean() * abs(conf[s].mean() - acc[s].mean()); print(f"  {a:.2f}-{b:.2f} | {s.sum():5d} | {conf[s].mean():.3f} | {acc[s].mean():.3f}")
print(f"ECE (15 bins) {ece:.4f}")
u = 1 - (P ** 2).sum(1); print(f"1 - sum p^2: mean {u.mean():.4f}, share of images with > .1: {(u > .1).mean():.3f}, > .5: {(u > .5).mean():.3f}")
