"""theory agent, step 6: amplitude-free checks of finding 1, a fair control for P4, and the nesting number.

(a) Match scores that do not depend on activation size:
    overlap = best share of the Base neuron's top-1% test rows that are also a Small neuron's top-1% rows (topsets)
    wmatch  = best write cosine to any Small neuron (writesR_b0; null best ~.205)
    Spearman with std, value, freq; partial on redundancy.
(b) P4 control: for orphans o, compare value(p*) with a look-alike q drawn only from counterpart neurons
    (Base neurons that are some Small neuron's best match), q closest in co-firing with o to p*.
(c) Nesting: share of p-on photos (peak > 0) where s is on, agreeing pairs (thresh.json data).
"""
import json
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr, rankdata

R = "../results/pair/"
d = np.load(R + "theory/stats.npz")
ts = np.load(R + "topsets.npz")
ids, nrows = ts["test_ids"], int(ts["test_nrows"])
K = ids.shape[1]
M = sp.csr_matrix((np.ones(ids.size, np.float32), (np.repeat(np.arange(7168), K), ids.ravel())), shape=(7168, nrows))
O = (M[3072:] @ M[:3072].T).toarray() / K  # Base x Small overlap of top-1% rows
overlap = O.max(1)
# control: overlap with Small after shuffling Small's rows (chance best overlap)
rng = np.random.default_rng(0); perm = rng.permutation(nrows)
Ms = sp.csr_matrix((np.ones(3072 * K, np.float32), (np.repeat(np.arange(3072), K), perm[ids[:3072].ravel()])), shape=(3072, nrows))
ov_null = (M[3072:] @ Ms.T).toarray().max(1) / K
Wr = np.load(R + "writesR_b0.npy").astype(np.float32)  # (3072, 4096)
wmatch = Wr.max(0)
rk = lambda x: rankdata(x) / len(x)
def partial(y, x, ctrl):
    X = np.column_stack([np.ones(len(y))] + [rk(c) for c in ctrl])
    ry = rk(y) - X @ np.linalg.lstsq(X, rk(y), rcond=None)[0]
    rx = rk(x) - X @ np.linalg.lstsq(X, rk(x), rcond=None)[0]
    return spearmanr(ry, rx)[0]
res = {"overlap_median": float(np.median(overlap)), "overlap_null_median": float(np.median(ov_null)),
       "wmatch_median": float(np.median(wmatch))}
print(res)
for name, y in (("corr match", d["test_match"]), ("overlap", overlap), ("write match", wmatch)):
    for k in ("test_std", "value", "freq", "wnorm"):
        r0 = spearmanr(y, d[k])[0]; r1 = partial(y, d[k], [d["test_redund"]])
        res[f"{name}|{k}"] = [float(r0), float(r1)]
        print(f"{name:12s} vs {k:9s} raw {r0:+.3f}  partial(redund) {r1:+.3f}")
# (b)
bs, sbb = d["test_best_s"], d["test_sbest_b"]
pstar = sbb[bs]; idx = np.arange(4096); orph = pstar != idx
cps = np.unique(sbb)
g = np.load(R + "grams.npz"); C = g["test_Cbb"].astype(np.float64); s_ = np.sqrt(np.diag(C)); Rbb = C / s_[:, None] / s_[None, :]
val = d["value"]; win_p = []; win_q = []
for o in np.where(orph)[0]:
    p = pstar[o]; c = cps[(cps != o) & (cps != p)]
    q = c[np.argmin(np.abs(Rbb[o, c] - Rbb[o, p]))]
    win_p.append(val[o] < val[p]); win_q.append(val[o] < val[q])
res["P4_value_o_lt_pstar"] = float(np.mean(win_p)); res["P4_control_o_lt_q_counterpart"] = float(np.mean(win_q))
print("P4: value(o)<value(p*)", np.mean(win_p), " control value(o)<value(q in counterparts, same co-firing)", np.mean(win_q))
res["orphan_share"] = float(orph.mean()); res["orphan_floor_by_count"] = 1 - len(cps) / 4096
print("orphan share", orph.mean(), "floor by count (1 - #counterparts/4096)", 1 - len(cps) / 4096)
# (c)
pk = np.load(R + "tools_peaks_3000.npz"); S = pk["small"][:, 0, :].astype(np.float32); B = pk["base"][:, 0, :].astype(np.float32)
th = json.load(open(R + "theory/thresh.json"))
nest = [((S[:, t["s"]] > 0) & (B[:, t["p"]] > 0)).sum() / max((B[:, t["p"]] > 0).sum(), 1) for t in th]
base_rate = [(S[:, t["s"]] > 0).mean() for t in th]
res["nesting_median"] = float(np.median(nest)); res["nesting_baserate_median"] = float(np.median(base_rate))
print("nesting: share of p-on photos where s is on, median", np.median(nest), " (s's base rate", np.median(base_rate), ")")
json.dump(res, open(R + "theory/checks.json", "w"), indent=1)
