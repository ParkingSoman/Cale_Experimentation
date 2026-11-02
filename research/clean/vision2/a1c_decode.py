"""A1b check 3 redone: ridge decode with the penalty chosen on held-out images; null match quality."""
import json, os, time
import numpy as np, mlx.core as mx
exec(open("a1_small.py").read().split("res = {}")[0])
A1 = json.load(open(os.path.join(RES, "vision-a1-small.json")))
img = {2: np.repeat(np.arange(len(CAL)), NPOS[2]), 3: np.repeat(np.arange(len(CAL)), NPOS[3])}
out = {}
for small in ("atto", "nano"):
    P3 = A1[small]["P3_pairs"]; s3 = []
    for q in P3:
        s = 3 if q["child"][0] >= 15 else 2; hp = HT[s][:, pos[s][tuple(q["parent"])]]; hn = HT[s][:, pos[s][tuple(q["null"])]]
        s3.append((q["corr_pc"], np.corrcoef(hp, hn)[0, 1]))
    s3 = np.array(s3); out[small] = {"corr_with_parent_family_null_median": np.nanmedian(s3, 0).tolist()}
    HS, _, _ = hidden(small)
    for s in (2, 3):
        st = f"stage{s + 1}"; d = np.log1p([desc[k] for k in ids[s]])
        tr, va, te = img[s] % 3 == 0, img[s] % 3 == 1, img[s] % 3 == 2
        A = np.c_[HS[s], np.ones(len(HS[s]))].astype(np.float64); T = HT[s].astype(np.float64); G0 = A[tr].T @ A[tr]; AT = A[tr].T @ T[tr]; sc = np.trace(G0) / len(G0)
        best = None
        for lam in (1e-1, 1, 10, 100):
            Wr = np.linalg.solve(G0 + np.eye(len(G0)) * lam * sc, AT); e = ((T[va] - A[va] @ Wr) ** 2).sum()
            if best is None or e < best[0]: best = (e, lam, Wr)
        P = A[te] @ best[2]; r2 = 1 - ((T[te] - P) ** 2).sum(0) / (((T[te] - T[te].mean(0)) ** 2).sum(0) + 1e-9)
        bins = {"leaf": d == 0, "1-2": (d > 0) & (d <= np.log1p(2)), "3-10": (d > np.log1p(2)) & (d <= np.log1p(10)), ">10": d > np.log1p(10)}
        out[small][st] = {"lam": best[1], "decode_r2_median_by_desc": {k: float(np.median(r2[v])) for k, v in bins.items()}}
        out[small][st + "_r2"] = r2.tolist()
    print(small, json.dumps({k: v for k, v in out[small].items() if not k.endswith("_r2")}), f"{time.time() - t0:.0f}s", flush=True); del HS
json.dump(out, open(os.path.join(RES, "vision-a1c-decode.json"), "w"))
