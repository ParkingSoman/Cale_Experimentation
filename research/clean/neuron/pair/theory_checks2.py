"""theory agent, step 7: is the overlap result a measurement artifact? Control = the same overlap score inside Base
(best top-1% overlap with ANOTHER Base neuron). If frequency/write norm predict Base-Base overlap equally, the effect is
the measure, not Small. Also: Small-vs-Base overlap minus Base-vs-Base overlap ('excess survival')."""
import json
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr, rankdata

R = "../results/pair/"
d = np.load(R + "theory/stats.npz")
ts = np.load(R + "topsets.npz"); ids, nrows = ts["test_ids"], int(ts["test_nrows"]); K = ids.shape[1]
M = sp.csr_matrix((np.ones(ids.size, np.float32), (np.repeat(np.arange(7168), K), ids.ravel())), shape=(7168, nrows))
Mb, Ms = M[3072:], M[:3072]
ov_bs = (Mb @ Ms.T).toarray().max(1) / K
Obb = (Mb @ Mb.T).toarray() / K; np.fill_diagonal(Obb, 0); ov_bb = Obb.max(1)
rk = lambda x: rankdata(x) / len(x)
def partial(y, x, ctrl):
    X = np.column_stack([np.ones(len(y))] + [rk(c) for c in ctrl])
    f = lambda v: rk(v) - X @ np.linalg.lstsq(X, rk(v), rcond=None)[0]
    return spearmanr(f(y), f(x))[0]
res = {"ov_bs_median": float(np.median(ov_bs)), "ov_bb_median": float(np.median(ov_bb)),
       "frac_bs_gt_bb": float((ov_bs > ov_bb).mean())}
print(res)
for k in ("freq", "wnorm", "test_std", "value"):
    a = spearmanr(ov_bs, d[k])[0]; b = spearmanr(ov_bb, d[k])[0]; c = partial(ov_bs, d[k], [ov_bb])
    e = spearmanr(ov_bs - ov_bb, d[k])[0]
    res[k] = dict(base_small=float(a), base_base_control=float(b), partial_ctrl_bb=float(c), excess=float(e))
    print(f"{k:9s} Base-Small {a:+.3f} | Base-Base control {b:+.3f} | partial on control {c:+.3f} | excess(bs-bb) {e:+.3f}")
# write-norm quintiles
q = np.quantile(d["wnorm"], np.linspace(0, 1, 6))
for i in range(5):
    s = (d["wnorm"] >= q[i]) & (d["wnorm"] <= q[i + 1])
    print(f"wnorm quintile {i}: overlap Base-Small {ov_bs[s].mean():.3f}  Base-Base {ov_bb[s].mean():.3f}  freq {d['freq'][s].mean():.3f}")
    res[f"wq{i}"] = [float(ov_bs[s].mean()), float(ov_bb[s].mean())]
print("spearman(freq, wnorm)", spearmanr(d["freq"], d["wnorm"])[0])
json.dump(res, open(R + "theory/checks2.json", "w"), indent=1)
