"""T3b: cross-model neuron matching by correlation (microscope check). Rules: LINEAGE-LOG.md T3b."""
import numpy as np
from common import *
from cores import big_blocks, preact

def zs(A):
    A = A - A.mean(0); s = A.std(0); s[s < 1e-9] = np.inf; return (A / s).astype(np.float32)

m, tok = load_pythia("160m"); B = big_blocks(tok); n = B.shape[0] * (SEQ - 1)
ch = {L: zs(gelu(preact(m, B, L).astype(np.float64))) for L in DEV_LAYERS}; del m
res = {L: {} for L in DEV_LAYERS}
for name in ("70m", "160m-seed1"):
    mp, _ = load_pythia(name)
    nl = len(layers(mp)); rng = np.random.default_rng(0)
    best = {L: np.zeros(3072) for L in DEV_LAYERS}; arg = {L: np.zeros(3072, int) for L in DEV_LAYERS}
    for pl in range(nl):
        P = zs(gelu(preact(mp, B, pl).astype(np.float64)))
        if name != "70m":   # equal candidate count: 12,288 of 36,864 -> keep 1/3 of each layer
            P = P[:, np.sort(rng.choice(P.shape[1], P.shape[1] // 3, replace=False))]
        for L in DEV_LAYERS:
            C = np.abs(ch[L].T @ P) / n
            b = C.max(1); upd = b > best[L]; best[L][upd] = b[upd]; arg[L][upd] = pl
    for L in DEV_LAYERS:
        res[L][name] = {"share_ge_775": float((best[L] >= .775).mean()), "share_ge_5": float((best[L] >= .5).mean()),
                        "median_best": float(np.median(best[L])), "parent_layer_counts": np.bincount(arg[L], minlength=nl).tolist()}
        print(L, name, res[L][name], flush=True)
    del mp
save("t3b", res)
