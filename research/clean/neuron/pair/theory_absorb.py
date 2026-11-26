"""theory agent, step 2: where does a low-match Base neuron's signal go in Small, and what picks the partner?

For each Base neuron o: s* = its best Small neuron (test corr). p* = the Base neuron that best matches s* (Small's
counterpart). For orphans (o != p*), compare the WRITE cosine and the CO-FIRING corr of (o, p*) with a look-alike
control q: the Base neuron whose co-firing corr with o is closest to corr(o, p*), q not in {o, p*}.
Also: value(o) vs value(p*).
"""
import numpy as np
from scipy.stats import rankdata

R = "../results/pair/"
d = np.load(R + "theory/stats.npz")
g = np.load(R + "grams.npz")
Cbb = g["test_Cbb"].astype(np.float64)
sb = np.sqrt(np.diag(Cbb)); Rbb = Cbb / sb[:, None] / sb[None, :]
V = np.load(R + "writes_V.npz")["b0"].astype(np.float64)
Vn = V / np.linalg.norm(V, axis=1, keepdims=True); Wbb = Vn @ Vn.T
n = 4096; idx = np.arange(n)
bs, sbest_b, m, val = d["test_best_s"], d["test_sbest_b"], d["test_match"], d["value"]
pstar = sbest_b[bs]
orph = pstar != idx
print(f"Base neurons that are not their best Small's counterpart: {orph.mean():.3f}")
rng = np.random.default_rng(0)
rows = []
for o in np.where(orph)[0]:
    p = pstar[o]
    c = Rbb[o].copy(); c[[o, p]] = np.inf
    q = np.argmin(np.abs(c - Rbb[o, p]))
    r = rng.integers(n)
    # percentile of write cosine and co-firing of p among all Base neurons (for o)
    wpct = (Wbb[o] < Wbb[o, p]).mean(); fpct = (Rbb[o] < Rbb[o, p]).mean()
    rows.append((m[o], Wbb[o, p], Wbb[o, q], Wbb[o, r], Rbb[o, p], Rbb[o, q], wpct, fpct, val[o] < val[p]))
A = np.array(rows, dtype=float)
lo = A[:, 0] < np.median(m)
for name, s in (("all orphans", slice(None)), ("orphans with match < median", lo), ("orphans with match >= median", ~lo)):
    a = A[s]
    print(f"{name} (n={len(a)}): write cos with p* {a[:,1].mean():.3f} | look-alike q {a[:,2].mean():.3f} | random {a[:,3].mean():.3f}"
          f" | p*>q in {(a[:,1]>a[:,2]).mean():.3f} | cofire p* {a[:,4].mean():.3f} q {a[:,5].mean():.3f}"
          f" | pct write {np.median(a[:,6]):.3f} pct cofire {np.median(a[:,7]):.3f} | value(o)<value(p*) {a[:,8].mean():.3f}")
np.save(R + "theory/absorb_rows.npy", A.astype(np.float32))
