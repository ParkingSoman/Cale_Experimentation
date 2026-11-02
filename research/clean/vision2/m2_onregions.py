"""M2: whole-network on-region containment, no groups. Rules: VISION-LOG.md M2."""
import json, time
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *

NB = 18; NULLN = 256; PPI = 49
m = load_convnext(); X, y = load_images(); ana, sea = split(y)
pick = lambda idx, n: np.concatenate([idx[y[idx] == c][:n] for c in range(1000)])
SETS = {"ana": pick(ana, 3), "sea": pick(sea, 1)}
H = [m.blocks[b].fc1.weight.shape[0] for b in range(NB)]; OFF = np.concatenate([[0], np.cumsum(H)])
pool7 = nn.MaxPool2d(7, 1, 3); rng = np.random.default_rng(0)


def maps(xb):
    x = m.stem_norm(m.stem(mx.array(norm_img(xb)))); Z, PRE = [], []
    for b in range(NB):
        if b > 0 and m.stage_of[b] != m.stage_of[b - 1]: x = m.down[m.stage_of[b] - 1](m.down_norm[m.stage_of[b] - 1](x))
        pre = m.blocks[b].pre(x); Z.append(m.blocks[b].fc1(pre)); PRE.append(pre); x = m.blocks[b](x)
    return Z, PRE


def to_res(z, pb, cb, view):
    k = 2 ** (m.stage_of[cb] - m.stage_of[pb])
    if k > 1: z = nn.MaxPool2d(k, k)(z)
    return pool7(z) if view else z


# null readouts: shuffled rows of random real neurons, bias set so the on-rate matches that neuron (from 200 images)
NULL = {}
Zs, Ps = maps(np.asarray(X[SETS["ana"][:200]]))
for b in range(1, NB):
    W = np.array(m.blocks[b].fc1.weight); bias = np.array(m.blocks[b].fc1.bias); src = rng.choice(H[b], NULLN, replace=False)
    Wn = np.stack([rng.permutation(W[s]) for s in src]); pre = np.array(Ps[b].reshape(-1, Ps[b].shape[-1])); zr = np.array(Zs[b].reshape(-1, H[b]))
    raw = pre @ Wn.T; rate = (zr[:, src] > 0).mean(0)
    bn = -np.array([np.quantile(raw[:, j], 1 - max(rate[j], 1e-4)) for j in range(NULLN)])
    NULL[b] = (mx.array(Wn.T), mx.array(bn.astype(np.float32)))
del Zs, Ps

t0 = time.time()
acc = {s: {} for s in SETS}
for s in SETS:
    for b in range(1, NB):
        P = int(OFF[b])
        acc[s][b] = {v: {"w": mx.zeros((H[b], P)), "u": mx.zeros((H[b], P)), "pon": mx.zeros((P,)),
                         "nw": mx.zeros((NULLN, P))} for v in ("view", "point")}
        acc[s][b].update({"won": mx.zeros((H[b],)), "on": mx.zeros((H[b],)), "rows": 0, "nwon": mx.zeros((NULLN,)), "non": mx.zeros((NULLN,))})
for s, idx in SETS.items():
    for i in range(0, len(idx), 25):
        Z, PRE = maps(np.asarray(X[idx[i:i + 25]])); B = Z[0].shape[0]
        for b in range(1, NB):
            _, Hh, Ww, _ = Z[b].shape; HW = Hh * Ww
            sel = mx.array(np.concatenate([j * HW + (np.arange(HW) if PPI >= HW else rng.choice(HW, PPI, replace=False)) for j in range(B)]))
            zc = Z[b].reshape(-1, H[b])[sel]; w = mx.maximum(zc, 0); on = (zc > 0).astype(mx.float32)
            pre = PRE[b].reshape(-1, PRE[b].shape[-1])[sel]; zn = pre @ NULL[b][0] + NULL[b][1]; wn = mx.maximum(zn, 0)
            a = acc[s][b]; a["won"] += w.sum(0); a["on"] += on.sum(0); a["rows"] += len(sel); a["nwon"] += wn.sum(0); a["non"] += (zn > 0).sum(0)
            for v in ("view", "point"):
                par = mx.concatenate([(to_res(Z[p], p, b, v == "view").reshape(-1, H[p])[sel] > 0).astype(mx.float32) for p in range(b)], 1)
                a[v]["w"] += w.T @ par; a[v]["u"] += on.T @ par; a[v]["pon"] += par.sum(0); a[v]["nw"] += wn.T @ par
            mx.eval(a)
        if i % 500 == 0: print(s, i, round(time.time() - t0), "s", flush=True)

blk_of = np.concatenate([[b] * H[b] for b in range(NB)])
out = {"blocks": {}}; save = {}
for v in ("view", "point"):
    out[v] = {}
    for b in range(1, NB):
        A, S = acc["ana"][b], acc["sea"][b]
        Cw = np.array(A[v]["w"]) / np.maximum(np.array(A["won"])[:, None], 1e-9); br = np.array(A[v]["pon"]) / A["rows"]
        rate = np.array(A["on"]) / A["rows"]; Cs = np.array(S[v]["w"]) / np.maximum(np.array(S["won"])[:, None], 1e-9)
        ok = (Cw >= .8) & (br[None] > rate[:, None]) & (br[None] <= .5)
        par = np.where(ok.any(1), np.argmin(np.where(ok, br[None], 9), 1), -1); has = par >= 0
        sealed = np.array([Cs[i, par[i]] >= .7 for i in np.where(has)[0]])
        Cn = np.array(A[v]["nw"]) / np.maximum(np.array(A["nwon"])[:, None], 1e-9); nrate = np.array(A["non"]) / A["rows"]
        okn = (Cn >= .8) & (br[None] > nrate[:, None]) & (br[None] <= .5)
        anyok = (Cw >= .8) & (br[None] > rate[:, None]); trivial = anyok.any(1) & ~has
        near = np.array([b - blk_of[np.where(ok[i])[0]].max() if has[i] else -1 for i in range(H[b])])
        o = {"coverage": float(has.mean()), "sealed_holds": float(sealed.mean()) if len(sealed) else None, "null_coverage": float(okn.any(1).mean()),
             "only_trivial_parent": float(trivial.mean()), "median_on_rate": float(np.median(rate)),
             "median_parent_breadth": float(np.median(br[par[has]])) if has.any() else None,
             "median_breadth_ratio": float(np.median(br[par[has]] / rate[has])) if has.any() else None,
             "nearest_parent_dist": {int(d): float((near == d).mean()) for d in np.unique(near[has])} if has.any() else {},
             "parent_block_hist": {int(k): int(c) for k, c in zip(*np.unique(blk_of[par[has]], return_counts=True))} if has.any() else {}}
        out[v][b] = o; save[f"{v}_{b}_par"] = par; save[f"{v}_{b}_rate"] = rate; save[f"{v}_{b}_Cbest"] = np.where(has, Cw[np.arange(H[b]), np.maximum(par, 0)], np.nan)
        print(f"[{v}] block {b:2d}: coverage {o['coverage']:.2f} (null {o['null_coverage']:.2f}) | sealed holds {o['sealed_holds']} | only-trivial {o['only_trivial_parent']:.2f} | "
              f"on-rate {o['median_on_rate']:.3f} parent breadth {o['median_parent_breadth']} ratio {o['median_breadth_ratio']} | nearest {o['nearest_parent_dist']}", flush=True)
np.savez_compressed(os.path.join(HF, "imagenet", "m2_parents.npz"), **save)
for v in ("view", "point"):
    cov = np.mean(np.concatenate([np.load(os.path.join(HF, "imagenet", "m2_parents.npz"))[f"{v}_{b}_par"] >= 0 for b in range(1, NB)]))
    print(f"[{v}] WHOLE NETWORK blocks 1-17: coverage {cov:.3f}"); out[v]["whole"] = float(cov)
json.dump(out, open(os.path.join(RES, "vision-m2-onregions.json"), "w"), indent=1)
