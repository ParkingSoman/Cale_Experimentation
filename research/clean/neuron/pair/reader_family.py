"""reader, step 4: which Base neurons carry the part s adds ("s on, p quiet" rows)? No model forward.

For each of the 30 pairs: rows A = s top 1% and p below its 90th percentile (196k rows, fwd.npz).
Each Base neuron b's ON rows: its topsets entries with value > 0 (the rest of a top-1% set is its most negative
rows, because GELU -> 0 from below). Enrichment = |A and top_b| / |A|.
The 20 most enriched Base neurons (p excluded) = "carriers". For them:
  read similarity to p: cosine of W1 rows in the u covariance metric (stream_cov Cuu), as a percentile among all Base
  co-firing with p: Jaccard of top-1% sets, as a percentile
Controls: (1) carriers of OTHER pairs' A rows, scored against THIS p (same procedure, wrong p);
          (2) carriers of THIS pair's p-top rows (s and p both on) - the upper reference.
Also: number of Base neurons (p excluded) a greedy union needs to cover 50% of A, of s's shared rows (s top and
p top), and of p's own on-rows (baseline: how many other Base neurons it takes to cover a Base neuron).
Output: results/pair/reader/family.json
"""
import json
import os

import numpy as np
from scipy import sparse

from cache import OUT

RD = os.path.join(OUT, "reader")


def pct(v, x):
    return float((v < x).mean() * 100)


def greedy_n(rows, B, excl, frac=0.5, cap=60):
    A = set(rows.tolist())
    U, n = set(), 0
    cols = B
    while len(U & A) < frac * len(A) and n < cap:
        Al = np.array(sorted(A - U))
        cnt = np.asarray(cols[Al].sum(0)).ravel()
        cnt[excl] = 0
        b = int(np.argmax(cnt))
        if cnt[b] == 0:
            break
        U |= set(Binc_T_rows[b])
        n += 1
    return n


def main():
    d = np.load(os.path.join(RD, "fwd.npz"))
    ts = np.load(os.path.join(OUT, "topsets.npz"))
    sc = np.load(os.path.join(OUT, "stream_cov.npz"))
    C, W1 = sc["Cuu"].astype(np.float64), sc["W1"].astype(np.float64)
    G = W1 @ C @ W1.T
    dg = np.sqrt(np.diag(G))
    RC = G / dg[:, None] / dg[None, :]
    np.fill_diagonal(RC, -np.inf)
    pairs = d["pairs"]
    SV, PV = d["SV"].astype(np.float32), d["PV"].astype(np.float32)
    N = SV.shape[0]
    K = N // 100
    nf = len(ts["fit_photos"]) * 49
    bi = np.concatenate([ts["fit_ids"][3072:], ts["test_ids"][3072:] + nf], 1)
    on = np.concatenate([ts["fit_vals"][3072:], ts["test_vals"][3072:]], 1).astype(np.float32) > 0  # on-rows only
    B = sparse.csr_matrix((on.ravel().astype(np.float32), (bi.ravel(), np.repeat(np.arange(4096), bi.shape[1]))),
                          shape=(N, 4096))
    B.eliminate_zeros()
    BtB = (B.T @ B).toarray()
    nb = np.diag(BtB).copy()
    JB = BtB / np.maximum(nb[:, None] + nb[None, :] - BtB, 1)
    np.fill_diagonal(JB, -np.inf)
    global Binc_T_rows
    Binc_T_rows = [r[o] for r, o in zip(bi, on)]
    A_rows, S_rows = [], []
    for k in range(len(pairs)):
        s, p = SV[:, k], PV[:, k]
        o = np.argsort(-s)[:K]
        quiet = p < np.percentile(p, 90)
        pt = np.zeros(N, bool); pt[np.argsort(-p)[:K]] = True
        A_rows.append(o[quiet[o]])
        S_rows.append(o[pt[o]])

    def carriers(rows, p, n=20):
        c = np.asarray(B[rows].sum(0)).ravel() / max(len(rows), 1)
        c[p] = -1
        return np.argsort(-c)[:n], c

    out, tab = [], []
    for k in range(len(pairs)):
        p = int(pairs[k, 1])
        car, c = carriers(A_rows[k], p)
        rd = [pct(RC[p][np.isfinite(RC[p])], RC[p, b]) for b in car]
        cf = [pct(JB[p][np.isfinite(JB[p])], JB[p, b]) for b in car]
        # control 1: other pairs' A carriers scored against this p
        c1r, c1f = [], []
        for j in range(len(pairs)):
            if j == k:
                continue
            carj, _ = carriers(A_rows[j], p)
            c1r += [pct(RC[p][np.isfinite(RC[p])], RC[p, b]) for b in carj if b != p]
            c1f += [pct(JB[p][np.isfinite(JB[p])], JB[p, b]) for b in carj if b != p]
        # reference: carriers of shared rows
        car2, _ = carriers(S_rows[k], p)
        rr = [pct(RC[p][np.isfinite(RC[p])], RC[p, b]) for b in car2]
        n50 = greedy_n(A_rows[k], B, p)
        n50s = greedy_n(S_rows[k], B, p) if len(S_rows[k]) > 20 else None
        pon = np.where(PV[:, k] > 0)[0]
        n50p = greedy_n(pon, B, p) if len(pon) > 20 else None
        row = dict(k=k, s=int(pairs[k, 0]), p=p, nA=int(len(A_rows[k])), top_carrier_cover=float(c[car[0]]),
                   carriers=car[:8].tolist(), carrier_cover=c[car[:8]].round(3).tolist(),
                   read_pct_med=float(np.median(rd)), cofire_pct_med=float(np.median(cf)),
                   ctrl_read_pct_med=float(np.median(c1r)), ctrl_cofire_pct_med=float(np.median(c1f)),
                   shared_read_pct_med=float(np.median(rr)), n_cover50_A=n50, n_cover50_shared=n50s,
                   n_cover50_p_on=n50p)
        out.append(row)
        tab.append([row["read_pct_med"], row["ctrl_read_pct_med"], row["cofire_pct_med"], row["ctrl_cofire_pct_med"],
                    row["shared_read_pct_med"], n50, n50s if n50s else np.nan, row["top_carrier_cover"],
                    n50p if n50p else np.nan])
        print(k, row["s"], p, "nA", row["nA"], "read %.0f (ctrl %.0f, shared %.0f)  cofire %.0f (ctrl %.0f)  n50 A %d shared %s p-on %s  top %.2f"
              % (row["read_pct_med"], row["ctrl_read_pct_med"], row["shared_read_pct_med"], row["cofire_pct_med"],
                 row["ctrl_cofire_pct_med"], n50, n50s, n50p, row["top_carrier_cover"]), flush=True)
    t = np.array(tab, float)
    summ = dict(zip(["read_pct", "ctrl_read_pct", "cofire_pct", "ctrl_cofire_pct", "shared_read_pct", "n50_A",
                     "n50_shared", "top_carrier_cover", "n50_p_on_rows"], np.nanmedian(t, 0).round(2).tolist()))
    print("median", summ)
    json.dump(dict(summary=summ, pairs=out), open(os.path.join(RD, "family.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
