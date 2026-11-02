"""S1: Fisher step1 write on M's read from different natural unsure subsets of equal size."""
import numpy as np, torch
import u9_write as U
from w_fisher import real_kl
from scipy.special import softmax
D, M = U.D, U.M; U.reads = {"M": M}
P = softmax(np.asarray(D.T[D.tr], np.float64), 1); u = 1 - (P ** 2).sum(1); uns = D.tr[u > .1]; ani = uns[D.y[uns] < 398]; non = uns[D.y[uns] >= 398]
print(f"unsure pool {len(uns)}: animal {len(ani)}, non-animal {len(non)}")
rng = np.random.default_rng(71); n = 4000; perm = rng.permutation(uns)
subsets = {"animal": np.sort(rng.choice(ani, min(n, len(ani)), replace=False)), "non-animal": np.sort(rng.choice(non, n, replace=False)),
           "randA": np.sort(perm[:n]), "randB": np.sort(perm[n:2 * n])}
B = {}
for k, idx in subsets.items():
    w0 = U.pass_(idx, seed=1); w1 = U.pass_(idx, base=w0, seed=2); B[k] = (M, *w1["M"]); print("done", k, len(idx), flush=True)
K = real_kl(U.m, U.fam, D, D.dv, B)
for k, v in K.items(): print(f"  {k:10s} dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})")
for a, b in [("animal", "non-animal"), ("randA", "randB"), ("animal", "randA"), ("non-animal", "randA")]:
    d = K[a] - K[b]; print(f"  paired {a} - {b}: {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})")
np.savez("cache/s1_mix.npz", **{k.replace('-', '_'): v for k, v in K.items()})
