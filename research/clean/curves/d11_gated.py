"""The 4a 1x1 neurons that training gates off with the loop signal: top natural crops (4a output channel j, intact net),
plus their response at loop centres vs elsewhere (intact)."""
import sys, numpy as np, torch
from inception import *
from family import Family, Data
from d4_write import stats
from study import sheet
m = load_model(); D = Data(seed=0, ndev=100); fam = list(np.load("cache/s1_fine.npz")["fam"])
ck = torch.load(f"cache/ckpt/{sys.argv[1]}.pt", map_location="cpu", weights_only=False); fm = Family(m, fam, 1, ks=ck.get("ks", 5)).to(DEV); fm.load_state_dict(ck["state"])
mu, C = stats(m, fm, fam, D, n=300)[0]
A = m.mixed4a_1x1_pre_relu_conv.weight.detach().cpu()[:, fam, 0, 0].double().numpy(); varI = np.einsum("oi,ij,oj->o", A, C[:10, :10], A)
t = fm.weights()[2][0][:, 0].cpu().numpy() * np.sqrt(C[10, 10]); score = t / np.sqrt(varI + 1)
gated = list(np.argsort(score)[:6]); print("most gated 4a 1x1 neurons:", gated, "trained col x sd(c) / sd(family input):", np.round(score[gated], 1))
idx = np.sort(np.random.default_rng(0).choice(D.tr, 1500, replace=False)); best = [[] for _ in gated]; corr = []
with torch.no_grad():
    for i in range(0, len(idx), 100):
        b = idx[i:i + 100]; ins = inputs_3b(m, to_3a(m, prep(D.X[b])))
        a4 = from_3b(m, F.relu(pre_3b(m, ins[0])), upto="4a")[:, gated]               # (B,6,14,14)
        c = F.max_pool2d(F.pad(F.relu(fm.pre(ins)), (0, 1, 0, 1), value=-1e9), 3, 2)[:, 0]
        corr.append((a4.permute(1, 0, 2, 3).reshape(6, -1).cpu().numpy(), c.reshape(-1).cpu().numpy()))
        v, p = a4.flatten(2).max(2)
        for k in range(6):
            for bi in range(len(b)): best[k].append((float(v[bi, k]), int(b[bi]), int(p[bi, k]) // 14 * 2, int(p[bi, k]) % 14 * 2))
A4 = np.concatenate([x[0] for x in corr], 1); Cc = np.concatenate([x[1] for x in corr])
print("intact corr(gated neuron, coarse loop signal):", np.round([np.corrcoef(A4[k], Cc)[0, 1] for k in range(6)], 3))
print("intact firing rate of gated neurons:", np.round((A4 > 0).mean(1), 3))
spots = [sorted(bk, reverse=True)[:10] for bk in best]
print(sheet(D.X, spots, "cache/sheet-gated-4a.png", size=96))
