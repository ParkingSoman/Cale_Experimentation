"""A1b: checks on A1. Rules: VISION-LOG.md A1b."""
import json, os, time
import numpy as np, mlx.core as mx
exec(open("a1_small.py").read().split("res = {}")[0])
A1 = json.load(open(os.path.join(RES, "vision-a1-small.json")))
rate = {s: (HT[s] > 1e-3).mean(0) for s in (2, 3)}
mt = load_convnext("tiny"); wn = {b: np.linalg.norm(np.array(mt.blocks[b].fc2.weight) * np.array(mt.blocks[b].gamma)[:, None], axis=0) for b in range(6, 18)}
def rk(v): return np.argsort(np.argsort(v)).astype(float)
img = {2: np.repeat(np.arange(len(CAL)), NPOS[2]), 3: np.repeat(np.arange(len(CAL)), NPOS[3])}
out = {}
for small in ("atto", "nano"):
    HS, _, _ = hidden(small); o = {}
    for s in (2, 3):
        st = f"stage{s + 1}"; r = np.array(A1[small]["best_match"][st]["r"]); arg = np.array(A1[small]["best_match"][st]["arg"])
        d = np.log1p([desc[k] for k in ids[s]]); blk = np.array([k[0] for k in ids[s]])
        en = np.log(HT[s].std(0) * np.array([wn[b][n] for b, n in ids[s]]) + 1e-9)
        Z = np.c_[np.eye(blk.max() + 1)[blk][:, sorted(set(blk))], rk(rate[s]), rk(en)]
        res_ = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]; a, bb = res_(rk(r)), res_(rk(d))
        o[f"{st}_P1_with_energy"] = float(a @ bb / np.linalg.norm(a) / np.linalg.norm(bb))
        o[f"{st}_spearman_desc_energy"] = float(np.corrcoef(rk(d), rk(en))[0, 1])
        # spread decode: ridge from all small neurons
        tr = img[s] % 2 == 0; te = ~tr; A = np.c_[HS[s], np.ones(len(HS[s]))].astype(np.float64)
        G = A[tr].T @ A[tr]; G += np.eye(len(G)) * 1e-2 * np.trace(G) / len(G)
        Wr = np.linalg.solve(G, A[tr].T @ HT[s][tr].astype(np.float64)); P = A[te] @ Wr; T = HT[s][te]
        r2 = 1 - ((T - P) ** 2).sum(0) / (((T - T.mean(0)) ** 2).sum(0) + 1e-9); del P, Wr, G
        bins = {"leaf": d == 0, "1-2": (d > 0) & (d <= np.log1p(2)), "3-10": (d > np.log1p(2)) & (d <= np.log1p(10)), ">10": d > np.log1p(10)}
        o[f"{st}_decode_r2_by_desc"] = {k: float(np.median(r2[v])) for k, v in bins.items()}
        o[f"{st}_single_r2_by_desc"] = {k: float(np.median(r[v] ** 2)) for k, v in bins.items()}
        a, bb = res_(rk(r2)), res_(rk(d)); o[f"{st}_P1_decode_partial"] = float(a @ bb / np.linalg.norm(a) / np.linalg.norm(bb))
        out.setdefault("r2", {})[f"{small}_{st}"] = r2.tolist()
    # null rate for shared best match
    P3 = A1[small]["P3_pairs"]; same_null = []
    for q in P3:
        s = 3 if q["child"][0] >= 15 else 2; st = f"stage{s + 1}"; arg = A1[small]["best_match"][st]["arg"]
        same_null.append(arg[pos[s][tuple(q["null"])]] == q["small"])
    o["child_shares_parent_match"] = float(np.mean([q["child_own_match_same"] for q in P3])); o["null_shares_parent_match"] = float(np.mean(same_null))
    out[small] = o; print(small, json.dumps(o), f"{time.time() - t0:.0f}s", flush=True); del HS
json.dump(out, open(os.path.join(RES, "vision-a1b-checks.json"), "w"))
