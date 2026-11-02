"""C3 (check only). Is 'the trained coarse kernel has 1.5% of its energy in the curve span' an artifact of the raw
metric? The rulers were trained WITHOUT weight decay, so kernel parts the input never excites keep random-init values.
Compare raw-metric and input-metric (Sigma = E[X X^T] over 5x5 windows of the bottleneck, 100 train images) span shares."""
import numpy as np, torch
from inception import *
from family import Data
m = load_model(); D = Data(); fam = list(np.load("cache/s1_fine.npz")["fam"])
idx = np.sort(np.random.default_rng(5).choice(D.tr, 100, replace=False)); S = torch.zeros(800, 800, dtype=torch.float64, device="cpu"); N = 0
with torch.no_grad():
    for i in range(0, 100, 20):
        c = inputs_3b(m, to_3a(m, prep(D.X[idx[i:i + 20]])))[2]
        U = F.unfold(F.pad(c, (2, 2, 2, 2)), 5).permute(0, 2, 1).reshape(-1, 800).cpu().double()
        S += U.T @ U; N += len(U)
S /= N; ev, Q = torch.linalg.eigh(S)
W5 = m.mixed3b_5x5_pre_relu_conv.weight.detach().cpu().double().reshape(96, 800)
F10 = W5[[i - 320 for i in fam]]
def share(k, B, M):   # M-metric share of k in span(B)
    G = B @ M @ B.T; a = torch.linalg.solve(G, B @ M @ k); p = B.T @ a
    return float((p @ M @ p) / (k @ M @ k))
I = torch.eye(800, dtype=torch.float64)
print("input metric: eigen spectrum share of top 50/100/200 dims: %.3f %.3f %.3f" % tuple(float(ev.flip(0)[:n].sum() / ev.sum()) for n in [50, 100, 200]))
for tag in ["L-k1-r-lr3e-2", "L-k1-m379-lr3e-2"]:
    st = torch.load(f"cache/ckpt/{tag}.pt", map_location="cpu")
    sd = st["state"] if "state" in st else st
    from family import Family
    fm = Family(m, fam, 1); fm.load_state_dict(sd); K = fm.weights()[0].cpu().double().reshape(800)
    null = Q[:, ev < ev.max() * 1e-4]; nf = float(((null.T @ K) ** 2).sum() / (K @ K))
    print(f"{tag}: raw share in curve span {share(K, F10, I):.3f} (all 96: {share(K, W5, I):.3f}) | input-metric share curve span "
          f"{share(K, F10, S):.3f} (all 96: {share(K, W5, S):.3f}) | raw energy in near-null input dims {nf:.3f}")
mean = F10.mean(0); print("family mean kernel: input-metric share of each member along it:",
      np.array([float((k @ S @ mean) ** 2 / ((k @ S @ k) * (mean @ S @ mean))) for k in F10]).round(3))
