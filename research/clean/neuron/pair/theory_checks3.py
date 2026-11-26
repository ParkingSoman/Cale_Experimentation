"""theory agent, step 8: tighten the FINAL.
(a) Is the ridge 'ceiling' limited by the sample? For 40 Base neurons, test R^2 of the best single other-Base neuron
    (chosen on fit) vs the 4,095-neuron ridge. Like-for-like floor: median match^2 vs median redundancy^2.
(b) Overlap using only FIRING rows (test_vals > 0) of each top-1% set: Base-Small best vs Base-Base best; partial.
(c) Which Small neurons are rebuilt well by the ridge? copy (p alone) or mix? What predicts it (Small-side stats)?
"""
import json
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr, rankdata

R = "../results/pair/"
d = np.load(R + "theory/stats.npz"); cj = json.load(open(R + "theory/ceiling.json")); lam = cj["lam"]
g = np.load(R + "grams.npz")
CF, CT = g["fit_Cbb"].astype(np.float64), g["test_Cbb"].astype(np.float64)
res = {}
# (a)
sF = np.sqrt(np.diag(CF)); RF = CF / sF[:, None] / sF[None, :]; np.fill_diagonal(RF, -9)
rng = np.random.default_rng(0); J = rng.choice(4096, 400, replace=False)[:40]
one = []
for j in J:
    k = RF[j].argmax(); a = CF[j, k] / CF[k, k]
    one.append(1 - (CT[j, j] - 2 * a * CT[j, k] + a * a * CT[k, k]) / CT[j, j])
res["base_best_single_other_r2_median"] = float(np.median(one))
res["match_sq_median"] = float(np.median(d["test_match"] ** 2)); res["redund_sq_median"] = float(np.median(d["test_redund"] ** 2))
print("(a) Base neuron from its best single other Base neuron, test R^2 median", np.median(one), "(ridge on 4095 gave .04)")
print("    like-for-like: median match^2", res["match_sq_median"], " median redundancy^2", res["redund_sq_median"])
# (b)
ts = np.load(R + "topsets.npz"); ids, vals, nrows = ts["test_ids"], ts["test_vals"].astype(np.float32), int(ts["test_nrows"])
K = ids.shape[1]; on = vals > 0
r_, c_ = np.nonzero(on)
M = sp.csr_matrix((np.ones(len(r_), np.float32), (r_, ids[r_, c_])), shape=(7168, nrows))
nfire = on.sum(1).astype(np.float64)
Mb, Ms = M[3072:], M[:3072]
I_bs = (Mb @ Ms.T).toarray(); I_bb = (Mb @ Mb.T).toarray(); np.fill_diagonal(I_bb, 0)
nb, ns = nfire[3072:], nfire[:3072]
J_bs = (I_bs / (nb[:, None] + ns[None, :] - I_bs + 1e-9)).max(1)
J_bb = (I_bb / (nb[:, None] + nb[None, :] - I_bb + 1e-9)).max(1)
ok = nb >= 20
print(f"(b) firing rows only; Base neurons with >=20 firing top rows: {ok.sum()}; median firing rows {np.median(nb):.0f}")
rk = lambda x: rankdata(x) / len(x)
def partial(y, x, ctrl):
    X = np.column_stack([np.ones(len(y))] + [rk(c) for c in ctrl])
    f = lambda v: rk(v) - X @ np.linalg.lstsq(X, rk(v), rcond=None)[0]
    return spearmanr(f(y), f(x))[0]
res["jac_bs_median"] = float(np.median(J_bs[ok])); res["jac_bb_median"] = float(np.median(J_bb[ok]))
print("    Jaccard best Small", res["jac_bs_median"], " best other Base", res["jac_bb_median"])
for k in ("test_std", "wnorm", "value", "freq"):
    a = spearmanr(J_bs[ok], d[k][ok])[0]; b = spearmanr(J_bb[ok], d[k][ok])[0]; c = partial(J_bs[ok], d[k][ok], [J_bb[ok]])
    c2 = partial(d["test_match"], d[k], [d["test_redund"]])
    res[f"jac|{k}"] = dict(bs=float(a), bb=float(b), partial=float(c))
    print(f"    {k:9s} Base-Small {a:+.3f} | Base-Base {b:+.3f} | partial on Base-Base {c:+.3f}")
# (c)
r2 = np.load(R + "theory/ceiling_r2.npy")
bas = json.load(open(R + "theory/basin.json"))["rows"]; alone = np.array([r["alone"] for r in bas])
for name, s in (("ridge R^2 > .5", r2 > .5), ("ridge R^2 .2-.5", (r2 > .2) & (r2 <= .5)), ("ridge R^2 < .2", r2 <= .2)):
    print(f"(c) {name}: n={s.sum()}, ridge {np.median(r2[s]):.3f}, p alone {np.median(alone[s]):.3f}, share by p {np.median(alone[s]/np.maximum(r2[s],1e-3)):.2f}")
    res[f"group {name}"] = dict(n=int(s.sum()), ridge=float(np.median(r2[s])), alone=float(np.median(alone[s])))
d = dict(d); d["svalue"] = d["test_sstd"] * d["swnorm"]
for k in ("test_sstd", "swnorm", "svalue", "sfreq", "test_sbest"):
    x = d[k]
    rr = spearmanr(r2, x)[0]; res[f"ridge|{k}"] = float(rr); print(f"    spearman(ridge R^2, Small {k}) {rr:+.3f}")
json.dump(res, open(R + "theory/checks3.json", "w"), indent=1)
