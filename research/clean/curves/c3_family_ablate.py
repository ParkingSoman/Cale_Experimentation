"""Damage of removing the curve family (10 neurons) from 3b: set to zero / to its mean. Also: random 10 neurons
of the 5x5 branch, and the 10 strongest non-curve 5x5 neurons, for scale. 2000 dev images."""
import numpy as np, torch
from inception import *
from crush import kl
m = load_model(); X, y = load_images(); tr, dv, se = split(y)
T = np.load("cache/teacher_logits.npy", mmap_mode="r"); lab = np.load("label_map.npy"); mu = torch.from_numpy(np.load("cache/mean3b.npy")).to(DEV)
rng = np.random.default_rng(1); dv2 = np.sort(rng.choice(dv, 2000, replace=False))
def run(sel, fill):
    s = torch.tensor(sel, device=DEV); ks, tp = [], []
    with torch.no_grad():
        for i in range(0, len(dv2), 100):
            b = dv2[i:i+100]; h = F.relu(pre_3b(m, to_3a(m, prep(X[b])))).clone()
            h[:, s] = 0 if fill == "zero" else mu[s][None, :, None, None]
            lg = from_3b(m, h); t = torch.from_numpy(T[b].astype(np.float32)).to(DEV)
            ks.append(kl(t, lg).cpu().numpy()); tp.append(lg[:, lab].argmax(1).cpu().numpy() == y[b])
    k = np.concatenate(ks); return f"KL {k.mean():.4f} (se {k.std()/np.sqrt(len(k)):.4f}) top-1 {np.concatenate(tp).mean():.4f}"
print("curves -> zero:", run(CURVES, "zero")); print("curves -> mean:", run(CURVES, "mean"))
others = [i for i in range(320, 416) if i not in CURVES]
for r in range(2): print("random 10 of 5x5 -> mean:", run(list(rng.choice(others, 10, replace=False)), "mean"))
