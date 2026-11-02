"""L1b: consolidation test with a well-posed fit (PCA-256 population). Rules: LINEAGE-LOG.md L1b."""
import os, sys
import numpy as np
from common import *

REVS = ["step0", "step64", "step256", "step512", "step1000", "step2000", "step4000", "step8000", "step16000", "step32000", "step64000", "main"]
STEP = {r: (143000 if r == "main" else int(r[4:])) for r in REVS}
D = os.path.join(HF, "lineage-data"); PER = 511
L = int(sys.argv[1]); assert L in DEV_LAYERS
tr = np.r_[0:3 * PER, 6 * PER:7 * PER]; te = np.r_[3 * PER:6 * PER, 7 * PER:8 * PER]
H = lambda rev: gelu(np.load(os.path.join(D, f"{rev}.npz"))[f"Z{L}"].astype(np.float64))


def stdz(A, ref):
    mu, sd = ref.mean(0), ref.std(0); sd[sd < 1e-12] = np.inf; return (A - mu) / sd


HT = H("main"); Ytr, Yte = stdz(HT[tr], HT[tr]), stdz(HT[te], HT[tr])
Ztr_T = stdz(HT[te], HT[te])
res = {"layer": L, "rows": []}
R2 = {}; SG = {}
for rev in REVS:
    Ht = H(rev)
    A, B = stdz(Ht[tr], Ht[tr]), stdz(Ht[te], Ht[tr])
    U, s, Vt = np.linalg.svd(A, full_matrices=False); V = Vt[:256].T
    Ptr, Pte = A @ V, B @ V
    G = Ptr.T @ Ptr; lam = 1e-3 * np.trace(G) / 256
    W = np.linalg.solve(G + lam * np.eye(256), Ptr.T @ Ytr)
    E = Yte - Pte @ W
    R2[rev] = 1 - (E ** 2).sum(0) / ((Yte - Yte.mean(0)) ** 2).sum(0)
    Bz = stdz(Ht[te], Ht[te])
    SG[rev] = (((Bz.T @ Ztr_T) / len(te)) ** 2).max(0)
    row = {"rev": rev, "pop_r2_median": float(np.median(R2[rev])), "single_corr2_median": float(np.median(SG[rev]))}
    res["rows"].append(row); print(row, flush=True)
tp = np.full(3072, np.inf); ts = np.full(3072, np.inf)
for rev in REVS[1:]:
    okp = (R2[rev] >= .6) & (R2[rev] >= R2["step0"] + .2)
    tp[okp & np.isinf(tp)] = STEP[rev]
    ts[(SG[rev] >= .6) & np.isinf(ts)] = STEP[rev]
share = float(((ts >= 4 * tp) & np.isfinite(tp)).mean())
res["B"] = {"share_tsingle_ge_4x_tpop": share, "reach_pop": float(np.isfinite(tp).mean()), "reach_single": float(np.isfinite(ts).mean()),
            "tpop_median": float(np.median(tp[np.isfinite(tp)])) if np.isfinite(tp).any() else None,
            "tsingle_median": float(np.median(ts[np.isfinite(ts)])), "step0_r2_median": float(np.median(R2["step0"])), "pass": share >= .3}
res["tpop"] = np.where(np.isfinite(tp), tp, -1).tolist(); res["tsingle"] = np.where(np.isfinite(ts), ts, -1).tolist()
print(res["B"])
save(f"l1b-L{L}", res)
