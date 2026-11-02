"""Card W2c: the Fisher write with enough images to converge (rule fixed before: fresh-image predicted damage within
10% of the solve-image value). Derived read only. Then paired dev-5000 against vmr and Bayes."""
import sys, json, numpy as np, torch
from inception import *
from family import Data
from d4_write import stats, writes
from w_fisher import load_build, solve, predict_and_real, real_kl
n = int(sys.argv[1]); n = min(n, 35000); tag = sys.argv[2] if len(sys.argv) > 2 else "derived2-r1-ks7"
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"])
fm = load_build(m, fam, tag); idx = np.sort(np.random.default_rng(12).choice(D.tr, min(n, len(D.tr)), replace=False))
cols, cbs, cov = solve(m, fam, D, fm, idx, 10, 1, bs=50, save=f"cache/fisher_ne_{tag}_{n}.pt")
torch.save(([c.cpu() for c in cols], [b.cpu() for b in cbs]), f"cache/fisher_write_{tag}_{n}.pt")
B = {"fisher": (fm, cols, cbs)}
fresh = np.setdiff1d(D.tr, idx)[:500] if len(np.setdiff1d(D.tr, idx)) >= 500 else np.sort(np.random.default_rng(5).choice(D.dv, 500, replace=False))
p1, _ = predict_and_real(m, fam, D, idx[:500], B, 10); p2, r2 = predict_and_real(m, fam, D, fresh, B, 10)
print(f"n={n}: predicted on solve images {p1['fisher'].mean():.4f}, on fresh {p2['fisher'].mean():.4f} (real {r2['fisher'].mean():.4f})", flush=True)
res, _ = writes(m, fam, stats(m, fm, fam, D))
for w in ["bayes", "vmr"]: B[w] = (fm, [c.to(DEV) for c in res[w][0]], res[w][1])
K = real_kl(m, fam, D, D.dv, B)
for k, v in K.items(): print(f"  {k:7s} dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})")
for b in ["vmr", "bayes"]:
    d = K["fisher"] - K[b]; print(f"  paired fisher - {b}: {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})")
json.dump({k: float(v.mean()) for k, v in K.items()} | dict(n=n, pred_solve=float(p1['fisher'].mean()), pred_fresh=float(p2['fisher'].mean())),
          open(f"../../../results/curves-w2c-{tag}-{n}.json", "w"), indent=1)
