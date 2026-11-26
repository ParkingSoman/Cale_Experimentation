"""reader, step 6: on-sets (pre-activation > 0). Does s keep p's positive pole and add to it? No model forward.
Per pair (fwd.npz, 196k rows): |s on| / |p on|; keep = fraction of p's on-rows where s is on (exact, and s on within
1 position, 3x3); add = fraction of s's on-rows where p is not on within 1 position.
Control: the same with other pairs' s. Also: fraction of A rows (s top 1%, p quiet) where ANY Base neuron is on
(topsets on-entries), against all rows. Output: results/pair/reader/onsets.json
"""
import json
import os

import numpy as np

from cache import OUT

RD = os.path.join(OUT, "reader")


def dil(m, NP):
    g = m.reshape(NP, 7, 7)
    o = g.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            sh = np.zeros_like(g)
            sh[:, max(-dy, 0):7 + min(-dy, 0), max(-dx, 0):7 + min(-dx, 0)] = g[:, max(dy, 0):7 + min(dy, 0), max(dx, 0):7 + min(dx, 0)]
            o |= sh
    return o.reshape(-1)


def main():
    d = np.load(os.path.join(RD, "fwd.npz"))
    ts = np.load(os.path.join(OUT, "topsets.npz"))
    SV, PV = d["SV"].astype(np.float32), d["PV"].astype(np.float32)
    N = SV.shape[0]
    NP = N // 49
    K = N // 100
    nf = len(ts["fit_photos"]) * 49
    bi = np.concatenate([ts["fit_ids"][3072:], ts["test_ids"][3072:] + nf], 1)
    on = np.concatenate([ts["fit_vals"][3072:], ts["test_vals"][3072:]], 1).astype(np.float32) > 0
    nB = np.bincount(bi[on], minlength=N)  # Base neurons on at each row (within their top 1%)
    S_on, P_on = SV > 0, PV > 0
    S_d = np.stack([dil(S_on[:, k], NP) for k in range(30)], 1)
    P_d = np.stack([dil(P_on[:, k], NP) for k in range(30)], 1)
    out, T = [], []
    for k in range(30):
        pk = P_on[:, k]
        keep = (S_on[pk, k]).mean() if pk.any() else np.nan
        keepd = (S_d[pk, k]).mean() if pk.any() else np.nan
        ctrl = np.nanmean([S_d[pk, j].mean() for j in range(30) if j != k]) if pk.any() else np.nan
        add = (~P_d[S_on[:, k], k]).mean()
        s, p = SV[:, k], PV[:, k]
        st = np.argsort(-s)[:K]
        A = st[(p < np.percentile(p, 90))[st]]
        r = dict(k=k, s=int(d["pairs"][k, 0]), p=int(d["pairs"][k, 1]), s_on=int(S_on[:, k].sum()), p_on=int(pk.sum()),
                 ratio=float(S_on[:, k].sum() / max(pk.sum(), 1)), keep=float(keep), keep_dil=float(keepd),
                 ctrl_keep_dil=float(ctrl), add=float(add), A_nBase_on=float(nB[A].mean()),
                 A_any_Base_on=float((nB[A] > 0).mean()), all_nBase_on=float(nB.mean()))
        out.append(r)
        T.append([r["ratio"], r["keep"], r["keep_dil"], r["ctrl_keep_dil"], r["add"], r["A_nBase_on"], r["all_nBase_on"],
                  r["A_any_Base_on"]])
        print(k, r, flush=True)
    names = ["ratio_s_on_over_p_on", "keep", "keep_dil", "ctrl_keep_dil", "add", "A_nBase_on", "all_rows_nBase_on",
             "A_any_Base_on"]
    summ = dict(zip(names, np.nanmedian(np.array(T, float), 0).round(3).tolist()))
    summ["any_Base_on_all_rows"] = float((nB > 0).mean())
    print(summ)
    json.dump(dict(summary=summ, pairs=out), open(os.path.join(RD, "onsets.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
