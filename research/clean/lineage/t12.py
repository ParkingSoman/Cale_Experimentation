"""T1 + T2: core identity and territory splits over Pythia-160M training. Rules: LINEAGE-LOG.md Round 2."""
import time
import numpy as np
from common import *
from cores import *

REVS = ["main", "step0", "step512", "step1000", "step2000", "step4000", "step8000", "step16000", "step32000", "step64000"]
STEP = {r: (143000 if r == "main" else int(r[4:])) for r in REVS}
_, tok = load_pythia(); B = big_blocks(tok); N = B.shape[0] * (SEQ - 1)
fin, out = {}, {L: [] for L in DEV_LAYERS}
for rev in REVS:
    t = time.time(); m, _ = load_pythia("160m", rev)
    for L in DEV_LAYERS:
        Z = preact(m, B, L)
        c50, c200 = core(Z, 50), core(Z, 200)
        if rev == "main":
            fin[L] = c50
        ident = (indicator(c50, N).multiply(indicator(fin[L], N))).sum(1).A.ravel() / 50
        st, anc, has, ins = split_stats(fin[L], coverage(fin[L], c200, N))
        self_anc = float(((anc == np.arange(3072)) & has).mean())
        row = {"rev": rev, "step": STEP[rev], "core_ident_median": float(np.median(ident)),
               "core_ident_q25_q75": [float(np.quantile(ident, .25)), float(np.quantile(ident, .75))], "self_ancestor_share": self_anc, **st}
        out[L].append(row); print(L, row, flush=True)
    del m; drop("160m", rev); print(rev, "%.0fs" % (time.time() - t), flush=True)
for L in DEV_LAYERS:
    r = {x["rev"]: x for x in out[L]}
    t1 = r["step8000"]["core_ident_median"] >= .5
    s0 = r["step0"]["split_share"]
    t2 = any(r[k]["split_share"] >= .10 and r[k]["split_share"] >= 3 * s0 for k in ("step1000", "step2000", "step4000", "step8000", "step16000"))
    save(f"t12-L{L}", {"layer": L, "rows": out[L], "T1_pass": bool(t1), "T2_pass": bool(t2)}); print(L, "T1", t1, "T2", t2)
