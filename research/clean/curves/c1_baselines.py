"""KL of simple replacements of 3b on 1000 dev images: zeros, per-channel mean, top-k neurons by energy (rest = mean)."""
import numpy as np, torch, json
from inception import *
from crush import kl, scales
m = load_model(); X, y = load_images(); tr, dv, se = split(y)
T = np.load("cache/teacher_logits.npy", mmap_mode="r"); lab = np.load("label_map.npy")
rng = np.random.default_rng(0); dv = np.sort(rng.choice(dv, 1000, replace=False)); trs = np.sort(rng.choice(tr, 2000, replace=False))
with torch.no_grad():
    S = torch.zeros(480, device=DEV); E = torch.zeros(480, device=DEV); n = 0
    for i in range(0, len(trs), 100):
        h = F.relu(pre_3b(m, to_3a(m, prep(X[trs[i:i+100]])))); S += h.sum((0, 2, 3)); E += (h**2).sum((0, 2, 3)); n += h.shape[0]*784
    mu = S / n; var = E / n - mu**2
    np.save("cache/mean3b.npy", mu.cpu().numpy()); np.save("cache/var3b.npy", var.cpu().numpy())
    order = torch.argsort(-var)  # energy around the mean
    def run(fn):
        ks, tp = [], []
        for i in range(0, len(dv), 100):
            b = dv[i:i+100]; h = F.relu(pre_3b(m, to_3a(m, prep(X[b])))); lg = from_3b(m, fn(h))
            t = torch.from_numpy(T[b].astype(np.float32)).to(DEV); ks.append(kl(t, lg).mean().item()); tp.append((lg[:, lab].argmax(1).cpu().numpy() == y[b]).mean())
        return round(float(np.mean(ks)), 4), round(float(np.mean(tp)), 4)
    print("intact", run(lambda h: h))
    print("zeros", run(lambda h: torch.zeros_like(h)))
    print("mean", run(lambda h: mu[None, :, None, None].expand_as(h).contiguous()))
    for k in [10, 20, 40, 120, 240]:
        keep = torch.zeros(480, dtype=torch.bool, device=DEV); keep[order[:k]] = True
        print(f"top-{k} by variance, rest mean", run(lambda h: torch.where(keep[None, :, None, None], h, mu[None, :, None, None])))
    print("curves only (10), rest mean", run(lambda h: torch.where(torch.isin(torch.arange(480, device=DEV), torch.tensor(CURVES, device=DEV))[None, :, None, None], h, mu[None, :, None, None])))
    print("scales", scales(m, X, trs[:64]))
