"""A1 P3 redone with a null matched on correlation with the parent (whole-block search) and firing rate."""
import json, os, time
import numpy as np, mlx.core as mx
exec(open("a1_small.py").read().split("res = {}")[0])
rate = {s: (HT[s] > 1e-3).mean(0) for s in (2, 3)}; ZT = {s: zs(HT[s]) for s in (2, 3)}
A1 = json.load(open(os.path.join(RES, "vision-a1-small.json"))); out = {}
blkcols = {s: {b: np.array([pos[s][(b, n)] for n in range(HTw[b])]) for b in colsT[s]} for s in (2, 3)}
for small in ("atto", "nano"):
    HS, _, _ = hidden(small); P3 = []; rr = np.random.default_rng(3)
    Q = A1[small]["P3_pairs"]; CP = {}
    for s in (2, 3):
        for b in colsT[s]:
            qs = [q for q in Q if q["child"][0] == b]
            if not qs: continue
            ps = sorted({pos[s][tuple(q["parent"])] for q in qs}); M = np.array(mx.matmul(mx.array(ZT[s][:, blkcols[s][b]]).T, mx.array(ZT[s][:, ps])))
            for j, jp in enumerate(ps): CP[(b, jp)] = M[:, j]
    for q in Q:
        b, n = q["child"]; p = tuple(q["parent"]); s = 3 if b >= 15 else 2; jp = pos[s][p]; jc = pos[s][(b, n)]
        cp = CP[(b, jp)]
        sub = subtree(p); ok = np.array([(b, k) not in sub and (b, k) != p for k in range(HTw[b])])
        rc = rate[s][jc]; rok = (rate[s][blkcols[s][b]] > rc / 2) & (rate[s][blkcols[s][b]] < rc * 2)
        cand = np.where(ok & rok)[0] if (ok & rok).sum() > 20 else np.where(ok)[0]
        k = int(cand[np.argmin(np.abs(cp[cand] - cp[n]))]); jn = blkcols[s][b][k]
        ss = HS[s][:, q["small"]]; hp = HT[s][:, jp]; A = np.c_[hp, np.ones(len(hp))]; resid = lambda v: v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
        pc = lambda v: float(np.corrcoef(resid(ss), resid(v))[0, 1]); arg = np.array(A1[small]["best_match"][f"stage{s + 1}"]["arg"])
        P3.append({"child": [b, n], "parent": list(p), "null": [b, k], "corr_p_child": float(cp[n]), "corr_p_null": float(cp[k]),
                   "pc_family": pc(HT[s][:, jc]), "pc_null": pc(HT[s][:, jn]), "share_family": bool(arg[jc] == q["small"]), "share_null": bool(arg[jn] == q["small"])})
    f = np.array([o["pc_family"] for o in P3]); nl = np.array([o["pc_null"] for o in P3]); d = f - nl
    bs = [np.nanmedian(d[rr.integers(0, len(d), len(d))]) for _ in range(1000)]
    sf = np.array([o["share_family"] for o in P3]); sn = np.array([o["share_null"] for o in P3])
    bs2 = [(lambda ix: sf[ix].mean() - sn[ix].mean())(rr.integers(0, len(sf), len(sf))) for _ in range(1000)]
    out[small] = {"n": len(P3), "corr_with_parent_median": [float(np.median([o["corr_p_child"] for o in P3])), float(np.median([o["corr_p_null"] for o in P3]))],
                  "pc_family": float(np.nanmedian(f)), "pc_null": float(np.nanmedian(nl)), "pc_diff": [float(np.nanmedian(d)), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                  "share_family": float(sf.mean()), "share_null": float(sn.mean()), "share_diff_ci": [float(np.quantile(bs2, .025)), float(np.quantile(bs2, .975))],
                  "by_stage": {st: {"pc_diff": float(np.nanmedian([o["pc_family"] - o["pc_null"] for o in P3 if (o["child"][0] >= 15) == (st == 4)])),
                                    "share": [float(np.mean([o["share_family"] for o in P3 if (o["child"][0] >= 15) == (st == 4)])), float(np.mean([o["share_null"] for o in P3 if (o["child"][0] >= 15) == (st == 4)]))]} for st in (3, 4)},
                  "pairs": P3}
    print(small, json.dumps({k: v for k, v in out[small].items() if k != "pairs"}), f"{time.time() - t0:.0f}s", flush=True); del HS
json.dump(out, open(os.path.join(RES, "vision-a1d-p3.json"), "w"))
