"""L1c: consolidation with enough text, computed online per checkpoint. Rules: LINEAGE-LOG.md L1c."""
import os, sys, time
import numpy as np
import mlx.core as mx
from common import *

REVS = ["main", "step0", "step512", "step1000", "step2000", "step4000", "step8000", "step16000", "step32000", "step64000"]
STEP = {r: (143000 if r == "main" else int(r[4:])) for r in REVS}
_, tok = load_pythia()
B = mx.concatenate([blocks(tok, "train", 40, skip=400), blocks(tok, "code_test", 8, skip=60)], 0)
nb = B.shape[0]; PER = SEQ - 1
blk = np.repeat(np.arange(nb), PER); tr, te = blk % 2 == 0, blk % 2 == 1


def stdz(A, ref):
    mu, sd = ref.mean(0), ref.std(0); sd[sd < 1e-12] = np.inf; return (A - mu) / sd


final, R2, SG = {}, {L: {} for L in DEV_LAYERS}, {L: {} for L in DEV_LAYERS}
for rev in REVS:
    t = time.time(); m, _ = load_pythia("160m", rev)
    for L in DEV_LAYERS:
        Wi, bi, _, _ = mlp_w(m, L)
        Ht = gelu((mlp_in(m, B, L).astype(np.float64)) @ Wi.T.astype(np.float64) + bi)
        if rev == "main":
            final[L] = (stdz(Ht[tr], Ht[tr]), stdz(Ht[te], Ht[tr]), stdz(Ht[te], Ht[te]))
        Ytr, Yte, Yz = final[L]
        A, Bt = stdz(Ht[tr], Ht[tr]), stdz(Ht[te], Ht[tr])
        G = A.T @ A; G += np.eye(3072) * 1e-3 * np.trace(G) / 3072
        W = np.linalg.solve(G, A.T @ Ytr)
        E = Yte - Bt @ W
        R2[L][rev] = 1 - (E ** 2).sum(0) / ((Yte - Yte.mean(0)) ** 2).sum(0)
        SG[L][rev] = (((stdz(Ht[te], Ht[te]).T @ Yz) / te.sum()) ** 2).max(0)
        print(rev, L, "pop R2 med %.3f" % np.median(R2[L][rev]), "single med %.3f" % np.median(SG[L][rev]), flush=True)
    del m; drop("160m", rev); print(rev, "%.0fs" % (time.time() - t), flush=True)

order = [r for r in sorted(REVS, key=lambda r: STEP[r]) if r != "step0"]
for L in DEV_LAYERS:
    tp = np.full(3072, np.inf); ts = np.full(3072, np.inf)
    for rev in order:
        okp = (R2[L][rev] >= .6) & (R2[L][rev] >= R2[L]["step0"] + .2)
        tp[okp & np.isinf(tp)] = STEP[rev]; ts[(SG[L][rev] >= .6) & np.isinf(ts)] = STEP[rev]
    gate = float(np.median(R2[L]["main"]))
    share = float(((ts >= 4 * tp) & np.isfinite(tp)).mean())
    res = {"layer": L, "gate_main_r2_median": gate, "gate_pass": gate >= .95,
           "rows": [{"rev": r, "step": STEP[r], "pop_r2_median": float(np.median(R2[L][r])), "pop_r2_q": [float(np.quantile(R2[L][r], q)) for q in (.25, .75)],
                     "single_corr2_median": float(np.median(SG[L][r]))} for r in sorted(REVS, key=lambda r: STEP[r])],
           "B": {"share": share, "pass": bool(share >= .3 and gate >= .95), "reach_pop": float(np.isfinite(tp).mean()),
                 "tpop_median": float(np.median(tp[np.isfinite(tp)])) if np.isfinite(tp).any() else None,
                 "tsingle_median": float(np.median(ts[np.isfinite(ts)])) if np.isfinite(ts).any() else None},
           "tpop": np.where(np.isfinite(tp), tp, -1).tolist(), "tsingle": np.where(np.isfinite(ts), ts, -1).tolist()}
    print(L, res["B"], "gate", gate)
    save(f"l1c-L{L}", res)
