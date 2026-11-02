"""R1 (E-small, NEURON-MATH Part IV): level-0 formula F(S-hat_n) for nested n; paired dev-5000. MIX-DERIVED."""
import numpy as np, torch
from inception import *
import b3_train
from family import Data, kl, pool3
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"])
tr = np.sort(np.random.default_rng(5).choice(D.tr, 2000, replace=False))     # same set as R3d
order = np.random.default_rng(11).permutation(tr)                            # nested subsets in this order
NS = [16, 64, 256, 1024, 2000]; S = torch.zeros(481, 481, dtype=torch.float64); saved = {}
with torch.no_grad():
    for i in range(0, 2000, 16):
        b = np.sort(order[i:i + 16]); x = pool3(F.relu(pre_3b(m, inputs_3b(m, to_3a(m, prep(D.X[b])))[0])))
        H = torch.cat([x.permute(0, 2, 3, 1).reshape(-1, 480), torch.ones(x.shape[0] * 196, 1, device=DEV)], 1).cpu().double()
        S += H.T @ H
        if i + 16 in NS: saved[i + 16] = S.clone()
def per_image(fm, idx, bs=100):
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]; lg = b3_train.forward_b3(m, fm, inputs_3b(m, to_3a(m, prep(D.X[b]))))
            out.append(kl(torch.from_numpy(D.T[b].astype(np.float32)).to(DEV), lg).cpu().numpy())
    return np.concatenate(out)
res = {n: per_image(b3_train.formula_k0(m, fam, saved[n]).to(DEV), D.dv) for n in NS}
np.savez("cache/x1_ladder.npz", **{str(n): v for n, v in res.items()})
ref = res[2000]
for n in NS:
    d = res[n] - ref
    print(f"n={n:5d}  KL {res[n].mean():.4f} (se {res[n].std() / np.sqrt(len(ref)):.4f})  minus n=2000 {d.mean():+.4f} (paired se {d.std() / np.sqrt(len(d)):.4f})", flush=True)
