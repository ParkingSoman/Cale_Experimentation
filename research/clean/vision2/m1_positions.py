"""M1: position-level territory map with view pooling, groups on what the child reads, shuffled-weight null.
Rules: VISION-LOG.md M1. Usage: python m1_positions.py 15,16,17 12"""
import sys, json, time
import numpy as np, mlx.core as mx, mlx.nn as nn
from sklearn.cluster import MiniBatchKMeans
from convnext import *

CH = [int(c) for c in sys.argv[1].split(",")]; P0 = int(sys.argv[2]); K = 2000; PPI = int(os.environ.get("PPI", "49"))
m = load_convnext(); X, y = load_images(); ana, sea = split(y)
pick = lambda idx, n: np.concatenate([idx[y[idx] == c][:n] for c in range(1000)])
SETS = {"ana": pick(ana, 5), "sea": pick(sea, 2)}
rng = np.random.default_rng(0)
NULLW = {}
for b in CH:
    W = np.array(m.blocks[b].fc1.weight); NULLW[b] = mx.array(np.stack([rng.permutation(r) for r in W]))
PARS = {b: list(range(P0, b)) for b in CH}
pool7 = nn.MaxPool2d(7, 1, 3)


def maps(xb):
    """z maps of all blocks P0..max(CH), and the child's read input."""
    x = m.stem_norm(m.stem(mx.array(norm_img(xb)))); Z, PRE = {}, {}
    for b in range(max(CH) + 1):
        if b > 0 and m.stage_of[b] != m.stage_of[b - 1]: x = m.down[m.stage_of[b] - 1](m.down_norm[m.stage_of[b] - 1](x))
        blk = m.blocks[b]
        if b >= P0:
            pre = blk.pre(x); Z[b] = blk.fc1(pre)
            if b in CH: PRE[b] = pre
        x = blk(x)
    return Z, PRE


def view(z, pb, cb):
    """parent z map -> child resolution, max over the child's view."""
    k = 2 ** (m.stage_of[cb] - m.stage_of[pb])
    if k > 1: z = nn.MaxPool2d(k, k)(z)
    return pool7(z)


def positions(t, sel):
    B, H, W, C = t.shape; return np.array(t.reshape(B * H * W, C))[sel]


# pass 1: k-means per child block on its read input
t0 = time.time(); KM = {}; samp = {b: [] for b in CH}
for i in range(0, 1500, 50):
    _, PRE = maps(np.asarray(X[SETS["ana"][i:i + 50]]))
    for b in CH: samp[b].append(np.array(PRE[b].reshape(-1, PRE[b].shape[-1])))
for b in CH:
    KM[b] = MiniBatchKMeans(K, batch_size=4096, n_init=1, random_state=0).fit(np.concatenate(samp[b]))
print("kmeans done", round(time.time() - t0), "s", flush=True)

# pass 2: group sums
acc = {s: {b: {"n": np.zeros(K), "z": 0, "null": 0, **{p: 0 for p in PARS[b]}} for b in CH} for s in SETS}
for s, idx in SETS.items():
    for i in range(0, len(idx), 50):
        Z, PRE = maps(np.asarray(X[idx[i:i + 50]]))
        for b in CH:
            B, H, W, C = PRE[b].shape; sel = np.arange(B * H * W)
            if PPI < H * W: sel = np.concatenate([j * H * W + rng.choice(H * W, PPI, replace=False) for j in range(B)])
            pre = positions(PRE[b], sel); g = KM[b].predict(pre); a = acc[s][b]
            oh = np.zeros((len(g), K), np.float32); oh[np.arange(len(g)), g] = 1
            a["n"] += oh.sum(0); a["z"] = a["z"] + oh.T @ positions(Z[b], sel)
            a["null"] = a["null"] + oh.T @ (pre @ np.array(NULLW[b]).T + np.array(m.blocks[b].fc1.bias))
            for p in PARS[b]: a[p] = a[p] + oh.T @ positions(view(Z[p], p, b), sel)
    print(s, "pass done", round(time.time() - t0), "s", flush=True)

means = {s: {b: {k: v / np.maximum(acc[s][b]["n"], 1)[:, None] for k, v in acc[s][b].items() if k != "n"} for b in CH} for s in SETS}
np.savez_compressed(os.path.join(HF, "imagenet", f"m1_means_{'-'.join(map(str, CH))}.npz"),
                    **{f"{s}_{b}_{k}": v for s in means for b in means[s] for k, v in means[s][b].items()},
                    **{f"{s}_{b}_n": acc[s][b]["n"] for s in acc for b in CH})


def pos_terr(v, valid):
    v = v[valid]; med = np.median(v, 0); return (v - med) / (v.max(0) - med + 1e-9) >= .5


def neg_terr(v, valid):
    v = v[valid]; med = np.median(v, 0); return (med - v) / (med - v.min(0) + 1e-9) >= .5


def cont(kid, par):
    k, p = kid.astype(np.float32), par.astype(np.float32); return (k.T @ p) / np.maximum(k.sum(0)[:, None], 1)


out = {}
for b in CH:
    M = means["ana"][b]; valid = acc["ana"][b]["n"] >= 5; o = {}
    for kind in ("z", "null"):
        kp, kn = pos_terr(M[kind], valid), neg_terr(M[kind], valid)
        res = {}
        for p in PARS[b]:
            pp = pos_terr(M[p], valid); br = pp.sum(0)
            cP = cont(kp, pp); cN = cont(kn, pp); broader = br[None] > kp.sum(0)[:, None]
            okP = (cP >= .8) & broader; okG = okP & (cN >= .8)
            res[p] = (okP.any(1), okG.any(1), np.where(okP, cP, -1).argmax(1), cP)
        prev = res[b - 1]; anyP = np.any([r[0] for r in res.values()], 0); anyG = np.any([r[1] for r in res.values()], 0)
        o[kind] = {"pos_prev": float(prev[0].mean()), "gate_prev": float(prev[1].mean()), "pos_any": float(anyP.mean()), "gate_any": float(anyG.mean()),
                   "median_breadth": float(np.median(kp.sum(0))), "median_neg_breadth": float(np.median(kn.sum(0)))}
        if kind == "z":
            Ms = means["sea"][b]; vs = valid & (acc["sea"][b]["n"] >= 2)
            kps = pos_terr(Ms["z"], vs); pps = pos_terr(Ms[b - 1], vs); cS = cont(kps, pps)
            hit = np.where(prev[0])[0]; par = prev[2][hit]
            d = np.abs(cS[hit, par] - prev[3][hit, par]); o[kind]["sealed_within_.1"] = float((d <= .1).mean())
            o[kind]["sealed_containment_median"] = float(np.median(cS[hit, par]))
            if b == 17:
                o["n2283_parent_prev"] = int(prev[2][2283]) if prev[0][2283] else None
    out[b] = o
    print(f"block {b}: real POS prev {o['z']['pos_prev']:.2f} any {o['z']['pos_any']:.2f} | GATE prev {o['z']['gate_prev']:.2f} any {o['z']['gate_any']:.2f} "
          f"|| null POS prev {o['null']['pos_prev']:.2f} any {o['null']['pos_any']:.2f} GATE any {o['null']['gate_any']:.2f} "
          f"|| breadth real {o['z']['median_breadth']:.0f} (neg {o['z']['median_neg_breadth']:.0f}) null {o['null']['median_breadth']:.0f} "
          f"|| sealed within .1: {o['z']['sealed_within_.1']:.2f} (median sealed containment {o['z']['sealed_containment_median']:.2f})", flush=True)
print("n2283 parent one block back:", out.get(17, {}).get("n2283_parent_prev"))
json.dump(out, open(os.path.join(RES, f"vision-m1-positions-{'-'.join(map(str, CH))}.json"), "w"), indent=1)
