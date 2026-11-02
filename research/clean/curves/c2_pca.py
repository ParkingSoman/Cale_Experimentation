"""KL when 3b output is replaced by its rank-k PCA reconstruction (positions as samples). Upper bound for a linear
k-dim code with a linear reader, ignoring that crushed neurons must be ReLU functions of the 3b inputs."""
import numpy as np, torch
from inception import *
from crush import kl
m = load_model(); X, y = load_images(); tr, dv, se = split(y)
T = np.load("cache/teacher_logits.npy", mmap_mode="r"); lab = np.load("label_map.npy")
rng = np.random.default_rng(0); dv = np.sort(rng.choice(dv, 500, replace=False)); trs = np.sort(rng.choice(tr, 2000, replace=False))
with torch.no_grad():
    C = torch.zeros(480, 480, device=DEV); S = torch.zeros(480, device=DEV); n = 0
    for i in range(0, len(trs), 100):
        h = F.relu(pre_3b(m, to_3a(m, prep(X[trs[i:i+100]])))).permute(0, 2, 3, 1).reshape(-1, 480)
        C += h.T @ h; S += h.sum(0); n += h.shape[0]
    mu = S / n; C = C / n - mu[:, None] * mu[None]
    ev, U = torch.linalg.eigh(C.cpu().double()); ev, U = ev.flip(0), U.flip(1)
    print("variance share top 10/20/40/120:", [round((ev[:k].sum() / ev.sum()).item(), 3) for k in [10, 20, 40, 120]])
    np.save("cache/pca3b.npy", U.numpy()); np.save("cache/pca3b_ev.npy", ev.numpy())
    for k in [10, 20, 40, 120, 240]:
        Uk = U[:, :k].float().to(DEV); ks, tp = [], []
        for i in range(0, len(dv), 100):
            b = dv[i:i+100]; h = F.relu(pre_3b(m, to_3a(m, prep(X[b]))))
            hc = (h - mu[None, :, None, None]).permute(0, 2, 3, 1); hr = (hc @ Uk @ Uk.T).permute(0, 3, 1, 2) + mu[None, :, None, None]
            lg = from_3b(m, hr); t = torch.from_numpy(T[b].astype(np.float32)).to(DEV)
            ks.append(kl(t, lg).mean().item()); tp.append((lg[:, lab].argmax(1).cpu().numpy() == y[b]).mean())
        print(f"PCA k={k}: KL {np.mean(ks):.3f} top-1 {np.mean(tp):.3f}")
