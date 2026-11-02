"""L1: lineage measures M1-M3 + key number on one dev layer. See LINEAGE-LOG.md L1 for the pre-registered rules."""
import os, sys, json
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from common import *

REVS = ["step0"] + [f"step{2**k}" for k in range(10)] + [f"step{s}" for s in (1000, 2000, 4000, 8000, 16000, 32000, 64000)] + ["main"]
STEP = {r: (143000 if r == "main" else int(r[4:])) for r in REVS}
D = os.path.join(HF, "lineage-data")
L = int(sys.argv[1]); assert L in DEV_LAYERS
NT = 4088; PER = 511
tr = np.r_[0:3 * PER, 6 * PER:7 * PER]; te = np.r_[3 * PER:6 * PER, 7 * PER:8 * PER]


def H(rev):
    return gelu(np.load(os.path.join(D, f"{rev}.npz"))[f"Z{L}"].astype(np.float64))


def zs(A):
    A = A - A.mean(0); s = A.std(0); s[s < 1e-12] = np.inf; return A / s


def corr(A):  # neurons x neurons
    Z = zs(A); return (Z.T @ Z) / A.shape[0]


HT = H("main"); ZT = zs(HT); ZTte = zs(HT[te])
out = {"layer": L, "rows": []}
first_pop = np.full(3072, np.inf); first_single = np.full(3072, np.inf)
ident = {}
S = {}
for rev in REVS:
    Ht = H(rev); Zt = zs(Ht)
    r = (Zt * ZT).mean(0)                         # M1
    ident[rev] = r
    C = corr(Ht); np.fill_diagonal(C, 0)          # M2
    S[rev] = float((C.max(1) >= 0.9).mean())
    # M3 held-out population R^2 and best single corr^2
    A = zs(Ht[tr]); Y = zs(HT[tr])
    G = A.T @ A; lam = 1e-2 * np.trace(G) / G.shape[0]
    W = np.linalg.solve(G + lam * np.eye(G.shape[0]), A.T @ Y)
    mu, sd = Ht[tr].mean(0), Ht[tr].std(0); sd[sd < 1e-12] = np.inf
    P = ((Ht[te] - mu) / sd) @ W
    Yte = (HT[te] - HT[tr].mean(0)) / np.where(HT[tr].std(0) < 1e-12, np.inf, HT[tr].std(0))
    r2 = 1 - ((Yte - P) ** 2).sum(0) / ((Yte - Yte.mean(0)) ** 2).sum(0)
    Cs = (zs(Ht[te]).T @ ZTte) / len(te)
    single = (Cs ** 2).max(0)
    st = STEP[rev]
    first_pop[(r2 >= 0.6) & np.isinf(first_pop)] = st
    first_single[(single >= 0.6) & np.isinf(first_single)] = st
    row = {"rev": rev, "step": st, "twin_share": S[rev], "ident_median": float(np.median(r)),
           "pop_r2_median": float(np.median(r2)), "single_corr2_median": float(np.median(single))}
    out["rows"].append(row); print(row, flush=True)

# birth step: first checkpoint from which identity stays >= 0.8
birth = np.full(3072, np.inf)
for k, rev in enumerate(REVS):
    ok = np.all([ident[r2_] >= 0.8 for r2_ in REVS[k:]], axis=0)
    birth[ok & np.isinf(birth)] = STEP[rev]
out["birth_quantiles"] = {q: float(np.quantile(birth[np.isfinite(birth)], q)) for q in (.1, .25, .5, .75, .9)} if np.isfinite(birth).any() else {}
out["birth_never_share"] = float(np.isinf(birth).mean())

# Reading A
early = [r for r in REVS if 1 <= STEP[r] <= 8000]
tstar = max(early, key=lambda r: S[r])
Ht = H(tstar); C = corr(Ht); np.fill_diagonal(C, 0)
iu = np.triu_indices(3072, 1)
tw = C[iu] >= 0.9
CT = corr(HT)
twin_final = CT[iu][tw]
A_pass = bool(S[tstar] >= 0.05 and S[tstar] >= 5 * max(S["step0"], S["main"]) and (len(twin_final) > 0 and np.median(twin_final) < 0.5))
out["A"] = {"tstar": tstar, "S_tstar": S[tstar], "S_step0": S["step0"], "S_end": S["main"],
            "n_twin_pairs": int(tw.sum()), "twin_final_corr_median": float(np.median(twin_final)) if len(twin_final) else None,
            "pass": A_pass}

# Reading B
both = np.isfinite(first_pop) & np.isfinite(first_single)
gap = np.where(both, first_single / np.maximum(first_pop, 1), np.nan)
share4 = float((gap >= 4).sum() / 3072)
out["B"] = {"share_tsingle_ge_4x_tpop": share4, "share_reach_pop": float(np.isfinite(first_pop).mean()),
            "share_reach_single": float(np.isfinite(first_single).mean()),
            "tpop_median": float(np.median(first_pop[np.isfinite(first_pop)])) if np.isfinite(first_pop).any() else None,
            "tsingle_median": float(np.median(first_single[np.isfinite(first_single)])) if np.isfinite(first_single).any() else None,
            "pass": share4 >= 0.30}

# Key number: groups at t* vs Ward on final input weights vs random, 768 groups each
def groups_from(Cm, method):
    Dm = np.clip(1 - np.abs(Cm), 0, 2); np.fill_diagonal(Dm, 0)
    return fcluster(linkage(squareform(Dm, checks=False), method), 768, "maxclust")


Wi = np.load(os.path.join(D, "main.npz"))[f"Wi{L}"].astype(np.float64)
Wn = Wi / np.linalg.norm(Wi, axis=1, keepdims=True)
Cw = Wn @ Wn.T
lin = groups_from(C + np.eye(3072), "average")
ward = fcluster(linkage(Wn, "ward"), 768, "maxclust")
rng = np.random.default_rng(0); rnd = rng.permutation(np.repeat(np.arange(768), 4))


def within(g, M):
    vals = []
    for k in np.unique(g):
        idx = np.where(g == k)[0]
        if len(idx) > 1:
            s = M[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]; vals.append(s)
    v = np.abs(np.concatenate(vals)) if vals else np.array([np.nan])
    sizes = np.bincount(g)[np.bincount(g) > 0]
    return {"n_pairs": int(len(v)), "mean": float(v.mean()),
            "max_group": int(sizes.max()), "singletons": int((sizes == 1).sum())}


out["key"] = {}
for name, g in (("lineage", lin), ("ward", ward), ("random", rnd)):
    out["key"][name] = {"final_fire_corr": within(g, CT), "final_in_cos": within(g, Cw)}
out["key"]["lineage_labels"] = lin.tolist()
print(json.dumps({k: v for k, v in out.items() if k not in ("rows",)}, default=str)[:3000])
save(f"l1-L{L}", out)
