"""S2: exact top-10 Fisher write (step0 + one active-set step) on M's read from N random unsure real images."""
import sys, time, numpy as np, torch
from inception import *
from family import Data
from w_fisher import load_build, solve, real_kl
from u4_w3 import solve_active_k
from scipy.special import softmax
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); M = load_build(m, fam, "L-k3-m-lr3e-2")
for p in m.parameters(): p.requires_grad_(False)
P = softmax(np.asarray(D.T[D.tr], np.float64), 1); uns = D.tr[1 - (P ** 2).sum(1) > .1]; rng = np.random.default_rng(81)
B = {}
for n in [int(a) for a in sys.argv[1:]]:
    idx = np.sort(rng.choice(uns, n, replace=False)); t0 = time.time()
    c0, b0, _ = solve(m, fam, D, M, idx, 10, 3); c1, b1, woke = solve_active_k(m, fam, D, M, idx, (c0, b0), 3)
    B[f"exact step1 {n}"] = (M, c1, b1); print(f"n {n}: {time.time() - t0:.0f} s, woken {woke:.4f}", flush=True)
K = real_kl(m, fam, D, D.dv, B)
for k, v in K.items(): print(f"  {k:18s} dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})")
