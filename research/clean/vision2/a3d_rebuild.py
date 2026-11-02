"""A3d: rebuild, lean memory, identity init (3 images per class) atto neurons from a tiny twin + its family. Rules: VISION-LOG.md A3."""
import json, os, time
import numpy as np, mlx.core as mx, mlx.nn as nn
from scipy.optimize import least_squares
from convnext import *
t0 = time.time(); X, y = load_images(); rank = np.zeros(len(y), int)
for c in range(1000): ix = np.where(y == c)[0]; rank[ix] = np.arange(len(ix))
CAL = np.where(rank < 3)[0]; rng = np.random.default_rng(0)
CFG = {"tiny": ((3, 3, 9, 3), (96, 192, 384, 768)), "atto": ((2, 2, 6, 2), (40, 80, 160, 320))}
NPOS = {2: 8, 3: 16}; SEL = {s: np.stack([rng.choice(g * g, NPOS[s], replace=False) for _ in CAL]) for s, g in ((2, 14), (3, 7))}
def pre_acts(name, only):
    m = load_convnext(name, *CFG[name]); ST = m.stage_of; out = {2: [], 3: []}; ids = {2: [], 3: []}
    for b in range(len(m.blocks)):
        if ST[b] >= 2: ids[ST[b]] += [(b, n) for n in range(m.blocks[b].fc1.weight.shape[0])]
    for i in range(0, len(CAL), 50):
        x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(X[CAL[i:i + 50]]))))); acc = {2: [], 3: []}
        for b in range(len(m.blocks)):
            if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
            if ST[b] == only:
                z = m.blocks[b].fc1(m.blocks[b].pre(x)); Bn, R, C, K = z.shape; s = ST[b]
                acc[s].append(np.array(z.reshape(-1, K)[mx.array((np.arange(Bn)[:, None] * R * C + SEL[s][i:i + Bn]).reshape(-1))]))
            x = m.blocks[b](x)
        out[only].append(np.concatenate(acc[only], 1))
        if ST[b] > only: pass
    return {only: np.concatenate(out[only]).astype(np.float32)}, ids
gelu = lambda z: z * 0.5 * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))
def zs(A): A = A - A.mean(0); return A / (np.linalg.norm(A, axis=0) + 1e-8)
MAP = json.load(open(os.path.join(ROOT, "research/clean/map/network_map_v2.json")))["neurons"]; kids = {}
for k, v in MAP.items(): b, n = map(int, k.split("_")); kids.setdefault(tuple(v["parent"]), []).append((b, n))
res = []
import resource, gc
for s in (2, 3):
    ZT, idT = pre_acts("tiny", s); ZA, idA = pre_acts("atto", s); print("acts stage", s + 1, ZT[s].shape, f"{time.time() - t0:.0f}s", flush=True)
    HTn = zs(gelu(ZT[s])); HAn = zs(gelu(ZA[s])); C = HAn.T @ HTn; posT = {k: j for j, k in enumerate(idT[s])}
    rkk = np.repeat(rank[CAL], NPOS[s]); tr, te = rkk <= 1, rkk == 2
    good = np.where(C.max(1) > .4)[0]; print("stage", s + 1, "atto neurons with r > .4:", len(good), "of", len(C), flush=True)
    for a in good:
        jt = int(C[a].argmax()); t = idT[s][jt]; hs = gelu(ZA[s][:, a]).astype(np.float64)
        ct = HTn.T @ HTn[:, jt] if False else None
        fam = []
        if f"{t[0]}_{t[1]}" in MAP and tuple(MAP[f"{t[0]}_{t[1]}"]["parent"]) in posT: fam.append(tuple(MAP[f"{t[0]}_{t[1]}"]["parent"]))
        ch = [c for c in kids.get(t, []) if c in posT]
        if ch:
            cc = [float(HTn[:, posT[c]] @ HTn[:, jt]) for c in ch]; fam += [ch[i] for i in np.argsort(cc)[::-1][:4]]
        if not fam: continue
        famset = set(fam) | {t} | set(kids.get(t, []))
        nul = []; ct_all = HTn.T @ HTn[:, jt]; blk_of = np.array([k[0] for k in idT[s]]); infam = np.array([k in famset for k in idT[s]])
        for f in fam:   # same block, not family, closest correlation with t
            jf = posT[f]; ok = (blk_of == f[0]) & ~infam
            for q in nul: ok[posT[q]] = False
            blkc = np.where(ok)[0]; nul.append(idT[s][blkc[int(np.argmin(np.abs(ct_all[blkc] - ct_all[jf])))]])
        def r2(pred): return float(1 - ((hs[te] - pred[te]) ** 2).sum() / ((hs[te] - hs[te].mean()) ** 2).sum())
        def lin(F):
            A = np.c_[F, np.ones(len(F))]; w = np.linalg.lstsq(A[tr], hs[tr], rcond=None)[0]; return A @ w
        def vote(Z):
            A = np.c_[Z, np.ones(len(Z))]; sd = A[tr].std(0) + 1e-6; sd[-1] = 1; A = A / sd
            g0 = gelu(Z[:, 0]); al, dd = np.linalg.lstsq(np.c_[g0, np.ones(len(g0))][tr], hs[tr], rcond=None)[0]
            w0 = np.zeros(A.shape[1]); w0[0] = sd[0]; p0 = np.r_[w0, al, dd]          # start AT the twin-only fit (identity init); f = lambda p: p[-2] * gelu(A[tr] @ p[:-2]) + p[-1] - hs[tr]
            sub = np.random.default_rng(0).choice(tr.sum(), min(8000, tr.sum()), replace=False)
            fs = lambda p: p[-2] * gelu(A[tr][sub] @ p[:-2]) + p[-1] - hs[tr][sub]
            p = least_squares(fs, p0, max_nfev=200).x; return p[-2] * gelu(A @ p[:-2]) + p[-1]
        cols = lambda L: [posT[k] for k in L]
        o = {"atto": [s, int(a)], "twin": list(t), "r": float(C[a, jt]), "family": [list(k) for k in fam], "null": [list(k) for k in nul],
             "S1": r2(lin(gelu(ZT[s][:, [jt]]).astype(np.float64))),
             "F_out": r2(lin(gelu(ZT[s][:, cols([t] + fam)]).astype(np.float64))), "N_out": r2(lin(gelu(ZT[s][:, cols([t] + nul)]).astype(np.float64))),
             "F_vote": r2(vote(ZT[s][:, cols([t] + fam)].astype(np.float64))), "N_vote": r2(vote(ZT[s][:, cols([t] + nul)].astype(np.float64))),
             "T_vote": r2(vote(ZT[s][:, [jt]].astype(np.float64)))}
        res.append(o)
    print("stage", s + 1, "done", len(res), f"{time.time() - t0:.0f}s", "maxrss GB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, flush=True)
    del ZT, ZA, HTn, HAn, C; gc.collect()
rr = np.random.default_rng(1); summ = {}
def ci(d): b = [np.median(d[rr.integers(0, len(d), len(d))]) for _ in range(1000)]; return [float(np.median(d)), float(np.quantile(b, .025)), float(np.quantile(b, .975))]
for st in (3, 4, "all"):
    R = [o for o in res if st == "all" or o["atto"][0] + 1 == st]
    if not R: continue
    g = lambda k: np.array([o[k] for o in R])
    summ[str(st)] = {"n": len(R), **{k: float(np.median(g(k))) for k in ("S1", "T_vote", "F_out", "N_out", "F_vote", "N_vote")},
                     "family_gain_minus_null_out": ci(g("F_out") - g("N_out")), "family_gain_minus_null_vote": ci(g("F_vote") - g("N_vote")),
                     "vote_minus_out_family": ci(g("F_vote") - g("F_out"))}
    print(st, json.dumps(summ[str(st)]), flush=True)
json.dump({"summary": summ, "rows": res}, open(os.path.join(RES, "vision-a3d-rebuild.json"), "w"))
print("done", f"{time.time() - t0:.0f}s")
