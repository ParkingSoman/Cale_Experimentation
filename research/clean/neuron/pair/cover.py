"""Pair study, firing sets: cover each Small neuron's top-1% rows with the fewest Base neurons' top-1% rows.

Input: topsets.npz. A_s = Small neuron s's top 1% rows; B_b = Base neuron b's top 1% rows (fit and test separately).
1. Greedy on FIT rows: add the Base neuron that most raises Jaccard(A_s, union of chosen B), up to 6, stop when no gain.
   Score on TEST rows: Jaccard, recall |A_s and U| / |A_s|, precision |A_s and U| / |U|.
2. Base-only relations of each added partner to term 1 (percentile among all Base neurons, 100 = closest):
   cofire  Jaccard of fit top sets      read  cosine of W1 rows      wout  cosine of stream writes   write  class effect
3. Rules that use Base only after term 1 is known: term 1 + its top m neighbours by each relation (m = 1, 2, 3).
   Controls: term 1 alone, term 1 + random Base neurons.
"""
import json
import os

import numpy as np
from scipy import sparse

from cache import OUT, load


def incidence(ids, nrows):
    n = ids.shape[0]
    r = ids.reshape(-1)
    c = np.repeat(np.arange(n), ids.shape[1])
    return sparse.csc_matrix((np.ones(len(r), np.float32), (r, c)), shape=(nrows, n))


def jac(a, u):
    inter = len(a & u)
    return inter / max(len(a | u), 1), inter / max(len(a), 1), inter / max(len(u), 1)


def main():
    d = np.load(os.path.join(OUT, "topsets.npz"))
    Mf = incidence(d["fit_ids"], int(d["fit_nrows"]))
    Mt_ids = d["test_ids"]
    f_ids = d["fit_ids"]
    Sf, Bf = Mf[:, :3072], Mf[:, 3072:].tocsr()
    sets_t = [set(x.tolist()) for x in Mt_ids]
    sets_f = [set(x.tolist()) for x in f_ids]
    O = (Sf.T @ Mf[:, 3072:]).toarray()           # (3072, 4096) fit overlaps
    BB = (Mf[:, 3072:].T @ Mf[:, 3072:]).toarray()  # Base co-firing counts
    K = f_ids.shape[1]
    Jbb = BB / (2 * K - BB)
    np.fill_diagonal(Jbb, -1)
    m = load("base")
    blk = m.features[7][0]
    W1 = blk.block[3].weight.detach().cpu().double().numpy()
    Wo = (blk.layer_scale.detach().cpu().double().view(-1, 1) * blk.block[5].weight.detach().cpu().double()).numpy().T
    Vw = np.load(os.path.join(OUT, "writes_V.npz"))["b0"].astype(np.float64)
    nrm = lambda A: A / np.linalg.norm(A, axis=1, keepdims=True).clip(1e-12)
    REL = dict(cofire=Jbb, read=nrm(W1) @ nrm(W1).T, wout=nrm(Wo) @ nrm(Wo).T, write=nrm(Vw) @ nrm(Vw).T)
    for k in ("read", "wout", "write"):
        np.fill_diagonal(REL[k], -np.inf)
    rng = np.random.default_rng(0)
    rows = []
    for s in range(3072):
        A = sets_f[s]
        chosen, U, best = [], set(), 0.0
        for _ in range(6):
            # Jaccard gain for each candidate: |A and (U or B)| / |A or U or B|
            gains = []
            cand = np.argsort(-O[s])[:50]  # candidates: largest raw overlap
            for b in cand:
                if b in chosen:
                    continue
                U2 = U | sets_f[3072 + b]
                gains.append((jac(A, U2)[0], b))
            j, b = max(gains)
            if j <= best + 1e-4:
                break
            chosen.append(int(b))
            U |= sets_f[3072 + b]
            best = j
        At = sets_t[s]
        tset = lambda bs: set().union(*[sets_t[3072 + b] for b in bs]) if bs else set()
        res = dict(s=s, terms=chosen, fit_jac=best)
        res["test"] = [jac(At, tset(chosen[:k])) for k in range(1, len(chosen) + 1)]
        t1 = chosen[0]
        res["partners"] = [{k: float((R[t1] < R[t1, b]).sum() / 4094 * 100) for k, R in REL.items()}
                           for b in chosen[1:4]]
        rules = {"t1": jac(At, tset([t1]))}
        for k, R in REL.items():
            nb = np.argsort(-R[t1])
            for mm in (1, 2, 3):
                rules[f"{k}_{mm}"] = jac(At, tset([t1] + nb[:mm].tolist()))
        for mm in (1, 2, 3):
            rules[f"rand_{mm}"] = jac(At, tset([t1] + rng.choice(4096, mm, replace=False).tolist()))
        oracle = res["test"]
        for mm in (1, 2, 3):
            rules[f"oracle_{mm}"] = oracle[min(mm, len(oracle) - 1)] if len(oracle) > 1 else oracle[0]
        res["rules"] = rules
        rows.append(res)
    summ = dict(n_terms_median=float(np.median([len(r["terms"]) for r in rows])),
                share_more_than_one_term=float(np.mean([len(r["terms"]) > 1 for r in rows])))
    for k in range(1, 5):
        v = [r["test"][min(k, len(r["test"])) - 1] for r in rows]
        summ[f"oracle_k{k}_test_jac_rec_prec"] = [float(np.median([x[i] for x in v])) for i in range(3)]
    P = [p for r in rows for p in r["partners"][:2]]
    for k in REL:
        summ[f"partner_pct_{k}_median"] = float(np.median([p[k] for p in P]))
        summ[f"partner_top1pct_{k}"] = float(np.mean([p[k] >= 99 for p in P]))
    for key in rows[0]["rules"]:
        summ[f"rule_{key}_jac_rec_prec"] = [float(np.median([r["rules"][key][i] for r in rows])) for i in range(3)]
    print(json.dumps(summ, indent=1))
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(OUT, "cover.json"), "w"))


if __name__ == "__main__":
    main()
