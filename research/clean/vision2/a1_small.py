"""A1: is a small ConvNeXt a collapsed tiny tree? Rules: VISION-LOG.md A1."""
import json, os, time
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
t0 = time.time(); X, y = load_images(); rank = np.zeros(len(y), int)
for c in range(1000): ix = np.where(y == c)[0]; rank[ix] = np.arange(len(ix))
CAL = np.where(rank < 1)[0]; rng = np.random.default_rng(0)
CFG = {"tiny": ((3, 3, 9, 3), (96, 192, 384, 768)), "atto": ((2, 2, 6, 2), (40, 80, 160, 320)), "nano": ((2, 2, 8, 2), (80, 160, 320, 640))}
NPOS = {2: 25, 3: 49}; SEL = {s: np.stack([rng.choice(g * g, NPOS[s], replace=False) for _ in CAL]) for s, g in ((2, 14), (3, 7))}

def hidden(name):
    m = load_convnext(name, *CFG[name]); ST = m.stage_of; out = {2: [], 3: []}; cols = {2: [], 3: []}
    for b in range(len(m.blocks)):
        if ST[b] >= 2: cols[ST[b]].append(b)
    for i in range(0, len(CAL), 50):
        x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(X[CAL[i:i + 50]]))))); acc = {2: [], 3: []}
        for b in range(len(m.blocks)):
            if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
            if ST[b] >= 2:
                h = nn.gelu(m.blocks[b].fc1(m.blocks[b].pre(x))); Bn, R, C, K = h.shape; s = ST[b]
                idx = mx.array((np.arange(Bn)[:, None] * R * C + SEL[s][i:i + Bn]).reshape(-1)); acc[s].append(np.array(h.reshape(-1, K)[idx]))
            x = m.blocks[b](x)
        for s in (2, 3): out[s].append(np.concatenate(acc[s], 1))
    return {s: np.concatenate(out[s]).astype(np.float32) for s in (2, 3)}, cols, [m.blocks[b].fc1.weight.shape[0] for b in range(len(m.blocks))]

def zs(A):
    A = A - A.mean(0); return A / (np.linalg.norm(A, axis=0) + 1e-8)

HT, colsT, HTw = hidden("tiny"); print("tiny", {s: v.shape for s, v in HT.items()}, f"{time.time() - t0:.0f}s", flush=True)
# tiny neuron id <-> (block, n)
ids = {s: [(b, n) for b in colsT[s] for n in range(HTw[b])] for s in (2, 3)}; pos = {s: {k: j for j, k in enumerate(ids[s])} for s in (2, 3)}
MAP = json.load(open(os.path.join(ROOT, "research/clean/map/network_map_v2.json")))["neurons"]
kids = {}
for k, v in MAP.items():
    b, n = map(int, k.split("_")); kids.setdefault(tuple(v["parent"]), []).append((b, n))
desc = {}
for b in range(17, -1, -1):
    for n in range(HTw[b]): desc[(b, n)] = sum(1 + desc[c] for c in kids.get((b, n), []))
def subtree(p):
    st, out = [p], set()
    while st:
        q = st.pop()
        for c in kids.get(q, []): out.add(c); st.append(c)
    return out
res = {}; rate = {s: (HT[s] > 1e-3).mean(0) for s in (2, 3)}
ZT = {s: zs(HT[s]) for s in (2, 3)}
for small in ("atto", "nano"):
    HS, _, _ = hidden(small); R = {}
    for s in (2, 3):
        Cm = np.array(mx.matmul(mx.array(ZT[s]).T, mx.array(zs(HS[s]))))
        R[s] = (Cm.max(1), Cm.argmax(1)); del Cm
    # P1: partial Spearman of best r on log(1+desc) given block and rate
    def rk(v): return np.argsort(np.argsort(v)).astype(float)
    P1 = {}
    for s in (2, 3):
        r = R[s][0]; d = np.log1p([desc[k] for k in ids[s]]); blk = np.array([k[0] for k in ids[s]])
        Z = np.c_[np.eye(blk.max() + 1)[blk][:, sorted(set(blk))], rk(rate[s])]
        res_ = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
        a, bb = res_(rk(r)), res_(rk(d)); rho = float(a @ bb / np.linalg.norm(a) / np.linalg.norm(bb))
        bs = []; rr = np.random.default_rng(1)
        for _ in range(300):
            ix = rr.integers(0, len(a), len(a)); bs.append(np.corrcoef(a[ix], bb[ix])[0, 1])
        bins = {"leaf": d == 0, "1-2": (d > 0) & (d <= np.log1p(2)), "3-10": (d > np.log1p(2)) & (d <= np.log1p(10)), ">10": d > np.log1p(10)}
        P1[s] = {"partial_spearman": [rho, float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                 "mean_r_by_desc": {k: [float(r[v].mean()), int(v.sum())] for k, v in bins.items()},
                 "mean_r_by_block": {int(b): float(r[blk == b].mean()) for b in sorted(set(blk))}}
        print(small, "stage", s + 1, json.dumps(P1[s]), flush=True)
    # P3: does p's best small match carry the child's split?
    P3 = []
    for s in (2, 3):
        Hs = HS[s]
        for (b, n) in ids[s]:
            p = tuple(MAP[f"{b}_{n}"]["parent"]) if f"{b}_{n}" in MAP else None
            if p is None or p not in pos[s] or rng.random() > (0.15 if s == 2 else 0.5): continue
            hp, hc = HT[s][:, pos[s][p]], HT[s][:, pos[s][(b, n)]]; ss = Hs[:, R[s][1][pos[s][p]]]
            A = np.c_[hp, np.ones(len(hp))]; resid = lambda v: v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
            pc = lambda v: float(np.corrcoef(resid(ss), resid(v))[0, 1])
            sub = subtree(p); cand = [j for j in rng.choice(HTw[b], 300, replace=False) if (b, int(j)) not in sub and (b, int(j)) != p]
            rc = rate[s][pos[s][(b, n)]]; cand = [j for j in cand if rc / 2 < rate[s][pos[s][(b, int(j))]] < rc * 2] or cand
            cpc = np.corrcoef(hp, hc)[0, 1]; cc = [abs(np.corrcoef(hp, HT[s][:, pos[s][(b, int(j))]])[0, 1] - cpc) for j in cand[:60]]
            j = int(cand[int(np.nanargmin(cc))])
            P3.append({"child": [b, n], "parent": list(p), "small": int(R[s][1][pos[s][p]]), "r_parent_small": float(R[s][0][pos[s][p]]),
                       "r_child_best": float(R[s][0][pos[s][(b, n)]]), "child_own_match_same": bool(R[s][1][pos[s][(b, n)]] == R[s][1][pos[s][p]]),
                       "pc_family": pc(hc), "pc_null": pc(HT[s][:, pos[s][(b, j)]]), "null": [b, j], "corr_pc": float(cpc)})
    f = np.array([o["pc_family"] for o in P3]); nl = np.array([o["pc_null"] for o in P3]); d = f - nl; rr = np.random.default_rng(2)
    bs = [np.median(d[rr.integers(0, len(d), len(d))]) for _ in range(1000)]
    summ = {"n": len(P3), "median_pc_family": float(np.nanmedian(f)), "median_pc_null": float(np.nanmedian(nl)),
            "median_diff": [float(np.nanmedian(d)), float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))],
            "child_shares_parent_match": float(np.mean([o["child_own_match_same"] for o in P3])),
            "by_stage": {st: float(np.nanmedian([o["pc_family"] - o["pc_null"] for o in P3 if (o["child"][0] >= 15) == (st == 4)])) for st in (3, 4)}}
    print(small, "P3", json.dumps(summ), f"{time.time() - t0:.0f}s", flush=True)
    res[small] = {"P1": {f"stage{s + 1}": v for s, v in P1.items()}, "P3": summ, "P3_pairs": P3,
                  "best_match": {f"stage{s + 1}": {"r": R[s][0].tolist(), "arg": R[s][1].tolist()} for s in (2, 3)}}
    del HS
json.dump({"ids": {f"stage{s + 1}": ids[s] for s in (2, 3)}, **res}, open(os.path.join(RES, "vision-a1-small.json"), "w"), default=float)
print("done", f"{time.time() - t0:.0f}s")
