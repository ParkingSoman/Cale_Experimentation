"""A1e: spread decode with every class in train and test. Rules: VISION-LOG.md A1e."""
import json, os, time
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
t0 = time.time(); X, y = load_images(); rank = np.zeros(len(y), int)
for c in range(1000): ix = np.where(y == c)[0]; rank[ix] = np.arange(len(ix))
IM = np.where(rank < 4)[0]; rng = np.random.default_rng(0)
CFG = {"tiny": ((3, 3, 9, 3), (96, 192, 384, 768)), "atto": ((2, 2, 6, 2), (40, 80, 160, 320)), "nano": ((2, 2, 8, 2), (80, 160, 320, 640))}
NP = 25; SEL = {s: np.stack([rng.choice(g * g, NP, replace=False) for _ in IM]) for s, g in ((2, 14), (3, 7))}
def hidden(name, keep=None):
    m = load_convnext(name, *CFG[name]); ST = m.stage_of; out = {2: [], 3: []}; ids = {2: [], 3: []}
    for b in range(len(m.blocks)):
        if ST[b] >= 2: ids[ST[b]] += [(b, n) for n in range(m.blocks[b].fc1.weight.shape[0])]
    for i in range(0, len(IM), 50):
        x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(X[IM[i:i + 50]]))))); acc = {2: [], 3: []}
        for b in range(len(m.blocks)):
            if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
            if ST[b] >= 2:
                h = nn.gelu(m.blocks[b].fc1(m.blocks[b].pre(x))); Bn, R, C, K = h.shape; s = ST[b]
                acc[s].append(np.array(h.reshape(-1, K)[mx.array((np.arange(Bn)[:, None] * R * C + SEL[s][i:i + Bn]).reshape(-1))]))
            x = m.blocks[b](x)
        for s in (2, 3):
            a = np.concatenate(acc[s], 1); out[s].append(a[:, keep[s]] if keep else a)
    return {s: np.concatenate(out[s]) for s in (2, 3)}, ids
mt_ids = hidden.__defaults__
_, idT = (None, None)
# ids of tiny without running twice: build from config
idT = {2: [(b, n) for b in range(6, 15) for n in range(1536)], 3: [(b, n) for b in range(15, 18) for n in range(3072)]}
keep = {s: np.sort(rng.choice(len(idT[s]), 2000, replace=False)) for s in (2, 3)}
HT, _ = hidden("tiny", keep); print("tiny", f"{time.time() - t0:.0f}s", flush=True)
MAP = json.load(open(os.path.join(ROOT, "research/clean/map/network_map_v2.json")))["neurons"]; kids = {}
for k, v in MAP.items(): b, n = map(int, k.split("_")); kids.setdefault(tuple(v["parent"]), []).append((b, n))
desc = {}
for b in range(17, -1, -1):
    for n in range(3072 if b >= 15 else 1536 if b >= 6 else 768 if b >= 3 else 384): desc[(b, n)] = sum(1 + desc[c] for c in kids.get((b, n), []))
rk = np.repeat(rank[IM], NP); tr, va, te = rk <= 1, rk == 2, rk == 3; out = {}
for small in ("atto", "nano"):
    HS, _ = hidden(small); out[small] = {}
    for s in (2, 3):
        st = f"stage{s + 1}"; d = np.array([desc[idT[s][j]] for j in keep[s]]); T = HT[s].astype(np.float64)
        A = np.c_[HS[s], np.ones(len(HS[s]))].astype(np.float64); G0 = A[tr].T @ A[tr]; AT = A[tr].T @ T[tr]; sc = np.trace(G0) / len(G0); best = None
        for lam in (1e-2, 1e-1, 1, 10):
            Wr = np.linalg.solve(G0 + np.eye(len(G0)) * lam * sc, AT); e = ((T[va] - A[va] @ Wr) ** 2).sum()
            if best is None or e < best[0]: best = (e, lam, Wr)
        P = A[te] @ best[2]; r2 = 1 - ((T[te] - P) ** 2).sum(0) / (((T[te] - T[te].mean(0)) ** 2).sum(0) + 1e-9)
        Zt = (T - T.mean(0)) / (T.std(0) + 1e-8); Zs = (HS[s] - HS[s].mean(0)) / (HS[s].std(0) + 1e-8); r1 = (Zt.T @ Zs / len(Zt)).max(1)
        bins = {"leaf": d == 0, "1-2": (d > 0) & (d <= 2), "3-10": (d > 2) & (d <= 10), ">10": d > 10}
        out[small][st] = {"lam": best[1], "decode_r2": {k: [float(np.median(r2[v])), int(v.sum())] for k, v in bins.items()},
                          "single_r": {k: float(np.median(r1[v])) for k, v in bins.items()}}
        print(small, st, json.dumps(out[small][st]), f"{time.time() - t0:.0f}s", flush=True)
    del HS
json.dump(out, open(os.path.join(RES, "vision-a1e-decode.json"), "w"))
