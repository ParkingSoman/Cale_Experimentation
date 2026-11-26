"""Pair study: where does each Base orphan go in Small?

For every Base neuron c (orphan or matched), on 3,000 photos (per-photo peaks, SEL half):
  host(c)  = the Small neuron, in any stage-4 block, whose peaks correlate most with c's.
Report for orphans against matched neurons: the host's block (same block as c, or earlier), host correlation,
and the relation between c and the host's own counterpart p* (the Base neuron the host is matched to):
  sibling   p* in the same block as c: cosine of their read rows (W1) and correlation of peaks
  parent    p* in an earlier block: virtual weight rank of c among p*'s children
Also, on c's top 1% photos (EVAL half): does the host fire more there than its counterpart p* does (both z-scored)?
That is the direct "the coarse neuron covers the orphan beyond its counterpart" test.
"""
import json
import os

import numpy as np

from cache import OUT, load
from tree import base_parts, vw


def z(a):
    return (a - a.mean(0)) / (a.std(0) + 1e-6)


def main():
    pk = np.load(os.path.join(OUT, "tools_peaks_3000.npz"))
    S = z(pk["small"].astype(np.float32))
    B = z(pk["base"].astype(np.float32))
    n = S.shape[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    sel = np.zeros(n, bool)
    sel[perm[: n // 2]] = True
    ev = ~sel
    Sf = S.reshape(n, -1)
    Bf = B.reshape(n, -1)
    Ss, Bs = z(Sf[sel]), z(Bf[sel])
    C = Bs.T @ Ss / sel.sum()  # (12288 base, 9216 small)
    host = C.argmax(1)
    hcorr = C.max(1)
    # counterparts: Small (block, s) -> Base p (agreeing argmax), and orphans
    cp = {}
    orph = set()
    for b in range(3):
        Rf = np.load(os.path.join(OUT, f"tools_peakR_b{b}.npy")).astype(np.float32)
        Rw = np.load(os.path.join(OUT, f"writesR_b{b}.npy")).astype(np.float32)
        af, aw = Rf.argmax(1), Rw.argmax(1)
        for s in np.where(af == aw)[0]:
            cp[b * 3072 + int(s)] = b * 4096 + int(af[s])
        orph |= {b * 4096 + c for c in set(range(4096)) - set(af.tolist()) - set(aw.tolist())}
    P = base_parts(load("base"))
    VW = {(k, k2): vw(P, k, k2).numpy() for k in (0, 1) for k2 in range(k + 1, 3)}
    W1 = [P[b]["W1"].numpy() for b in range(3)]
    W1n = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in W1]
    rows = []
    for c in range(12288):
        bc, cc = c // 4096, c % 4096
        h = int(host[c])
        hb = h // 3072
        r = dict(c=c, orphan=c in orph, host_block=hb, same_block=hb == bc, host_corr=float(hcorr[c]))
        ps = cp.get(h)
        if ps is not None:
            pb, pc = ps // 4096, ps % 4096
            r["host_has_counterpart"] = True
            r["counterpart_is_c"] = ps == c
            r["corr_c_pstar"] = float(Bs[:, c] @ Bs[:, ps] / sel.sum())
            if pb == bc and ps != c:
                r["relation"] = "sibling"
                r["read_cos"] = float(W1n[bc][cc] @ W1n[bc][pc])
            elif pb < bc:
                r["relation"] = "parent"
                order = np.argsort(-VW[(pb, bc)][pc])
                r["vw_rank"] = int(np.where(order == cc)[0][0])
            elif ps != c:
                r["relation"] = "later"
            top = np.where(ev)[0][np.argsort(-Bf[ev, c])[: max(ev.sum() // 100, 10)]]
            r["host_minus_pstar_on_c_top"] = float(Sf[top, h].mean() - Bf[top, ps].mean())
        else:
            r["host_has_counterpart"] = False
        rows.append(r)

    def summ(rs):
        out = dict(n=len(rs), host_corr_median=float(np.median([r["host_corr"] for r in rs])),
                   same_block=float(np.mean([r["same_block"] for r in rs])),
                   host_has_counterpart=float(np.mean([r["host_has_counterpart"] for r in rs])))
        hc = [r for r in rs if r["host_has_counterpart"]]
        out["counterpart_is_c"] = float(np.mean([r.get("counterpart_is_c", False) for r in hc]))
        for rel in ("sibling", "parent", "later"):
            out[f"rel_{rel}"] = float(np.mean([r.get("relation") == rel for r in hc]))
        sib = [r for r in hc if r.get("relation") == "sibling"]
        out["sibling_read_cos_median"] = float(np.median([r["read_cos"] for r in sib])) if sib else None
        out["sibling_corr_c_pstar_median"] = float(np.median([r["corr_c_pstar"] for r in sib])) if sib else None
        par = [r for r in hc if r.get("relation") == "parent"]
        out["parent_vw_rank_median"] = float(np.median([r["vw_rank"] for r in par])) if par else None
        d = [r["host_minus_pstar_on_c_top"] for r in hc if not r.get("counterpart_is_c")]
        out["host_minus_pstar_on_c_top_mean"] = float(np.mean(d)) if d else None
        out["host_above_pstar_share"] = float(np.mean(np.array(d) > 0)) if d else None
        return out

    O = [r for r in rows if r["orphan"]]
    M = [r for r in rows if not r["orphan"]]
    # baseline for read cosine: random same-block pairs
    rc = [float(W1n[b][i] @ W1n[b][j]) for b in range(3) for i, j in rng.integers(0, 4096, (2000, 2))]
    res = dict(orphans=summ(O), matched=summ(M), random_read_cos_median=float(np.median(rc)),
               random_read_cos_q99=float(np.quantile(rc, .99)))
    for k, v in res.items():
        print(k, v, flush=True)
    json.dump(dict(summary=res, rows=rows), open(os.path.join(OUT, "orphans.json"), "w"))


if __name__ == "__main__":
    main()
