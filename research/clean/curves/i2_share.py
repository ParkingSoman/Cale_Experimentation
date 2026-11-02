"""I2 + I3: Fisher step1 write on M's read, from subsets chosen by the network's own uncertainty u = 1 - sum p^2."""
import numpy as np, torch
import u9_write as U
from inception import *
from w_fisher import real_kl
from scipy.special import softmax
D, M = U.D, U.M; U.reads = {"M": M}
P = softmax(np.asarray(D.T[D.tr], np.float64), 1); u = 1 - (P ** 2).sum(1); rng = np.random.default_rng(31)
pick = lambda pool, n: np.sort(rng.choice(pool, n, replace=False))
lo, hi = D.tr[u < np.median(u)], D.tr[u > .1]; print(f"pool sizes: low {len(lo)}, high {len(hi)}; mean u: all {u.mean():.3f}, high {u[u > .1].mean():.3f}")
subsets = {"nat-4000": pick(D.tr, 4000), "low-4000": pick(lo, 4000), "high-4000": pick(hi, 4000), "high-1000": pick(hi, 1000),
           "high-10500": np.sort(hi), "nat-12000": pick(D.tr, 12000)}
B = {}
for name, idx in subsets.items():
    w0 = U.pass_(idx, seed=1); w1 = U.pass_(idx, base=w0, seed=2); B[name] = (M, *w1["M"]); print("done", name, flush=True)
K = real_kl(U.m, U.fam, D, D.dv, B); ref = np.load("cache/u9_kl_35000.npz")
for n, v in K.items():
    d = v - K["nat-4000"]; print(f"  {n:11s} dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f}) | paired vs nat-4000 {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})", flush=True)
print("u9 35000 file keys:", list(ref.keys()))
np.savez("cache/i2_share.npz", **{n.replace('-', '_'): v for n, v in K.items()})
