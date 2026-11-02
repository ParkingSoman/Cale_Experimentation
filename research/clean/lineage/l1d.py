"""L1d: are final FAMILIES (coarse) present early, when single neurons are not? Rules: LINEAGE-LOG.md L1d."""
import os, sys, time
import numpy as np
import mlx.core as mx
from scipy.cluster.hierarchy import linkage, fcluster
from common import *

REVS = ["main", "step0", "step1000", "step2000", "step4000", "step8000", "step16000", "step32000", "step64000"]
STEP = {r: (143000 if r == "main" else int(r[4:])) for r in REVS}
_, tok = load_pythia()
B = mx.concatenate([blocks(tok, "train", 40, skip=400), blocks(tok, "code_test", 8, skip=60)], 0)
blk = np.repeat(np.arange(B.shape[0]), SEQ - 1); tr, te = blk % 2 == 0, blk % 2 == 1


def stdz(A, ref):
    mu, sd = ref.mean(0), ref.std(0); sd[sd < 1e-12] = np.inf; return (A - mu) / sd


def r2(A, Bt, Ytr, Yte):
    G = A.T @ A; G += np.eye(G.shape[0]) * 1e-3 * np.trace(G) / G.shape[0]
    W = np.linalg.solve(G, A.T @ Ytr); E = Yte - Bt @ W
    return 1 - (E ** 2).sum(0) / ((Yte - Yte.mean(0)) ** 2).sum(0)


tg, res = {}, {L: [] for L in DEV_LAYERS}
for rev in REVS:
    t = time.time(); m, _ = load_pythia("160m", rev)
    for L in DEV_LAYERS:
        Wi, bi, _, _ = mlp_w(m, L)
        Ht = gelu(mlp_in(m, B, L).astype(np.float64) @ Wi.T.astype(np.float64) + bi)
        if rev == "main":
            Z = stdz(Ht, Ht[tr]); Z[~np.isfinite(Z)] = 0
            fam = fcluster(linkage(Z[tr].T, "ward"), 768, "maxclust") - 1
            rnd = np.random.default_rng(0).permutation(fam)
            M = lambda g: np.stack([Z[:, g == k].sum(1) for k in range(768)], 1)
            Yf, Yr = M(fam), M(rnd)
            Yall = np.c_[Z, Yf, Yr]
            tg[L] = (Yall[tr], Yall[te], np.bincount(fam))
        Ytr, Yte, sizes = tg[L]
        A, Bt = stdz(Ht[tr], Ht[tr]), stdz(Ht[te], Ht[tr]); A[~np.isfinite(A)] = 0; Bt[~np.isfinite(Bt)] = 0
        R = r2(A, Bt, Ytr, Yte)
        single, famr, rndr = R[:3072], R[3072:3072 + 768], R[3072 + 768:]
        multi = sizes >= 2
        row = {"rev": rev, "step": STEP[rev], "single": float(np.median(single)), "family": float(np.median(famr[multi])),
               "random_group": float(np.median(rndr[multi])), "fam_minus_rand": float(np.median(famr[multi]) - np.median(rndr[multi])),
               "family_size_median": float(np.median(sizes[multi]))}
        res[L].append(row); print(L, row, flush=True)
    del m; drop("160m", rev); print(rev, "%.0fs" % (time.time() - t), flush=True)
for L in DEV_LAYERS:
    ok = any(r["fam_minus_rand"] >= .15 for r in res[L] if r["step"] in (2000, 4000, 8000))
    save(f"l1d-L{L}", {"layer": L, "rows": res[L], "pass": ok}); print(L, "pass", ok)
