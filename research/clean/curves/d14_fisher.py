"""Fisher metric of the output on the ten family channels at 4a's input (after max-pool): G = E[ sum_p g(p) g(p)^T ],
g = d log p(y) / d x4a[family, p], y sampled from the intact softmax. Score each k-subset by tr(G Sigma_res), the
Fisher-weighted error left after the Bayes estimate of all ten from the kept ones (lower = better)."""
import sys, itertools, numpy as np, torch
from inception import *
from family import Data, IN4A, pool3, mixed
from d12_k3 import member_stats
m = load_model(); D = Data(seed=0, ndev=10); fam = list(np.load("cache/s1_fine.npz")["fam"]); k = int(sys.argv[1])
idx = np.sort(np.random.default_rng(41).choice(D.tr, 400, replace=False)); G = np.zeros((10, 10)); g_ = torch.Generator().manual_seed(0)
for i in range(0, len(idx), 25):
    with torch.no_grad(): x = pool3(F.relu(pre_3b(m, to_3a(m, prep(D.X[idx[i:i + 25]])))))
    x = x.clone().requires_grad_(True)
    h = mixed(m, "mixed4a", x)
    for n in ["mixed4b", "mixed4c", "mixed4d", "mixed4e"]: h = mixed(m, n, h)
    h = pool3(h)
    for n in ["mixed5a", "mixed5b"]: h = mixed(m, n, h)
    lg = m.softmax2_pre_activation_matmul(F.avg_pool2d(h, 7).reshape(-1, 1024)); lp = F.log_softmax(lg, 1)
    yb = torch.multinomial(lp.exp().detach().cpu(), 1, generator=g_)[:, 0].to(DEV)
    gr, = torch.autograd.grad(lp[torch.arange(len(yb)), yb].sum(), x)
    gf = gr[:, fam].permute(0, 2, 3, 1).reshape(-1, 10).cpu().numpy().astype(np.float64); G += gf.T @ gf
G /= len(idx); st = member_stats(m, fam, D); C = st[0][1]
np.set_printoptions(precision=2, suppress=True, linewidth=160)
print("Fisher diag (relative):", np.diag(G) / np.diag(G).mean()); w, V = np.linalg.eigh(G); print("Fisher eigen shares:", (w / w.sum())[::-1][:5])
pref = dict(zip(fam, [10, 50, 70, 120, 170, 180, 250, 290, 320, 350]))
def loss(K):
    K = [fam.index(f) for f in K]; R = C - C[:, K] @ np.linalg.solve(C[np.ix_(K, K)], C[K, :]); return np.trace(G @ R) / np.trace(G @ C)
rows = sorted([(loss(S), S) for S in itertools.combinations(fam, k)])
for s, S in rows[:6]: print(f"{s:.3f}", S, [pref[f] for f in S])
print("worst:", [(round(s, 3), [pref[f] for f in S]) for s, S in rows[-3:]])
for S, kl in [((330, 388, 385), .133), ((340, 343, 406), .164), ((343, 342, 385), .159), ((330, 340, 324), .208)]:
    if len(S) == k: print("tested", [pref[f] for f in S], f"Fisher loss {loss(S):.3f}  dev KL {kl}")
np.save("cache/fisher_family.npy", G)
