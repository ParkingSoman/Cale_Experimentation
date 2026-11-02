"""Choose kept members from statistics alone: for each k-subset, the share of the family's reader input that the
kept channels carry (own columns + Bayes estimate of the removed ones), summed over all 368 readers."""
import sys, itertools, numpy as np, torch
from inception import *
from family import Data, IN4A
from d12_k3 import member_stats
k = int(sys.argv[1]); m = load_model(); D = Data(seed=0, ndev=10); fam = list(np.load("cache/s1_fine.npz")["fam"])
st = member_stats(m, fam, D); As = [getattr(m, n).weight.detach().cpu()[:, fam, 0, 0].double().numpy() for n in IN4A]
pref = dict(zip(fam, [10, 50, 70, 120, 170, 180, 250, 290, 320, 350]))
def score(K):
    tot = 0.0; expl = 0.0
    for j, A in enumerate(As):
        mu, C = st[1] if j == 3 else st[0]
        tot += np.einsum("oi,ij,oj->", A, C, A)
        # best linear estimate of the full family input from kept channels: var explained = a C_K^-1 a^T, a = A C[:, K]
        a = A @ C[:, K]; expl += np.einsum("ok,kl,ol->", a, np.linalg.inv(C[np.ix_(K, K)]), a)
    return expl / tot
rows = sorted([(score([fam.index(f) for f in S]), S) for S in itertools.combinations(fam, k)], reverse=True)
for s, S in rows[:6]: print(f"{s:.3f}", S, [pref[f] for f in S])
print("...")
for s, S in rows[-3:]: print(f"{s:.3f}", S, [pref[f] for f in S])
for S in [(330, 388, 385), (340, 343, 406), (343, 342, 385), (330, 340, 324)]:
    if len(S) == k: print("tested", S, f"{score([fam.index(f) for f in S]):.3f}")
