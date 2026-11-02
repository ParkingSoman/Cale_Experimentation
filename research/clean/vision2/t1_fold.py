"""T1: composable tree fold with family parents. Rules: VISION-LOG.md T1. Usage: python t1_fold.py A|B|C"""
import sys, json, os, glob, time
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *

EXP = sys.argv[1]; t0 = time.time()
m = load_convnext(); X, y = load_images(); ST = m.stage_of; NB = 18
H = [m.blocks[b].fc1.weight.shape[0] for b in range(NB)]
rank = np.zeros(len(y), int)
for c in range(1000):
    ix = np.where(y == c)[0]; rank[ix] = np.arange(len(ix))
CAL = np.where(rank < 1)[0]; EV = np.where((rank >= 20) & (rank < 23))[0]
M3 = np.load(os.path.join(HF, "imagenet", "m3_matrices.npz")); RATE = {b: M3[f"{b}_rate"] for b in range(1, 18)}
MAP = json.load(open(os.path.join(ROOT, "research/clean/map/network_map_v2.json")))["neurons"]
G = [np.array(m.blocks[b].gamma, np.float64) for b in range(NB)]
W0 = [np.array(m.blocks[b].fc2.weight, np.float64) * G[b][:, None] for b in range(NB)]     # write columns (C, H)
B0 = [np.array(m.blocks[b].fc2.bias, np.float64) * G[b] for b in range(NB)]
F2 = [m.blocks[b].fc2 for b in range(NB)]
rng = np.random.default_rng(0)

# ---- calibration pass: hidden maps of all blocks on CAL images, 1/sigma at each downsample norm
SD = [0.0, 0.0, 0.0]; nsd = 0


def hidden_maps(xb):
    global nsd
    x = m.stem_norm(m.stem(mx.array(norm_img(xb)))); hs = []
    for b in range(NB):
        if b > 0 and ST[b] != ST[b - 1]:
            t = ST[b] - 1; SD[t] += float(mx.mean(1 / mx.sqrt(mx.var(x, axis=-1) + 1e-6)))
            x = m.down[t](m.down_norm[t](x))
        hs.append(nn.gelu(m.blocks[b].fc1(m.blocks[b].pre(x)))); x = m.blocks[b](x)
    return hs


def at_res(h, pb, cb):
    k = 2 ** (ST[cb] - ST[pb]); return nn.AvgPool2d(k, k)(h) if k > 1 else h


def collect(pairs):
    """pairs: list of (cb, cn, pb, pn). Returns per child-block: child cols, parent cols at the child's resolution."""
    global nsd
    need = {}
    for cb, cn, pb, pn in pairs:
        d = need.setdefault(cb, {"c": set(), "p": set()}); d["c"].add(cn); d["p"].add((pb, pn))
    need = {cb: {"c": sorted(d["c"]), "p": sorted(d["p"])} for cb, d in need.items()}
    acc = {cb: {"c": [], "p": []} for cb in need}
    for i in range(0, len(CAL), 50):
        hs = hidden_maps(np.asarray(X[CAL[i:i + 50]])); nsd += 1
        for cb, d in need.items():
            Bn, R, C, _ = hs[cb].shape; sel = np.concatenate([j * R * C + rng.choice(R * C, min(49, R * C), replace=False) for j in range(Bn)])
            acc[cb]["c"].append(np.array(hs[cb].reshape(-1, H[cb])[mx.array(sel)][:, mx.array(d["c"])]))
            cols = []
            for pb in sorted({p for p, _ in d["p"]}):
                ns = [pn for p, pn in d["p"] if p == pb]
                cols.append(np.array(at_res(hs[pb], pb, cb).reshape(-1, H[pb])[mx.array(sel)][:, mx.array(ns)]))
            acc[cb]["p"].append(np.concatenate(cols, 1))
    out = {}
    for cb, d in need.items():
        Hc = np.concatenate(acc[cb]["c"]).astype(np.float64); Hp = np.concatenate(acc[cb]["p"]).astype(np.float64)
        out[cb] = (d["c"], d["p"], Hc, Hp)
    return out


DSUM = [np.array(m.down[t].weight, np.float64).sum((1, 2)) for t in range(3)]; DNW = [np.array(m.down_norm[t].weight, np.float64) for t in range(3)]


def down_map(t):
    C = DSUM[t].shape[1]; P = np.eye(C) - 1.0 / C; return DSUM[t] @ np.diag(DNW[t]) @ P * (SD[t] / max(nsd, 1))


def back(v, sc, sp):
    """child write in stage sc -> parent-stage vector whose forward map best matches it (least squares)."""
    for t in range(sc - 1, sp - 1, -1):
        v = np.linalg.lstsq(down_map(t), v, rcond=None)[0]
    return v


# ---- evaluation
def logp():
    lg = np.concatenate([np.array(m(mx.array(norm_img(np.asarray(X[EV[i:i + 100]]))))) for i in range(0, len(EV), 100)]).astype(np.float64)
    mxv = lg.max(1, keepdims=True); return lg - mxv - np.log(np.exp(lg - mxv).sum(1, keepdims=True))


def evaluate(W, Bv):
    for b in range(NB):
        if W[b] is W0[b] and Bv[b] is B0[b]: continue
        f = nn.Linear(H[b], W0[b].shape[0]); f.weight = mx.array((W[b] / G[b][:, None]).astype(np.float32)); f.bias = mx.array((Bv[b] / G[b]).astype(np.float32)); m.blocks[b].fc2 = f
    lp = logp()
    for b in range(NB): m.blocks[b].fc2 = F2[b]
    return lp


def apply(pairs, fit, mode):
    """pairs (cb, cn, pb, pn), processed deepest child block first. mode: cut | fold."""
    W = [w.copy() for w in W0]; Bv = [b.copy() for b in B0]
    for cb, cn, pb, pn in sorted(pairs, key=lambda q: -q[0]):
        a, bb, mu = fit[(cb, cn, pb, pn)]; wc = W[cb][:, cn].copy(); W[cb][:, cn] = 0
        if mode == "cut": Bv[cb] += mu * wc; continue
        if ST[pb] != ST[cb]: Bv[cb] += mu * wc; continue          # cross-stage fold is invalid (T1 diag): cut instead
        Bv[cb] += bb * wc; W[pb][:, pn] += a * wc
    return W, Bv


def fits(pairs):
    data = collect(pairs); out = {}
    for cb, cn, pb, pn in pairs:
        cs, ps, Hc, Hp = data[cb]; hc = Hc[:, cs.index(cn)]; hp = Hp[:, ps.index((pb, pn))]
        A = np.c_[hp, np.ones(len(hp))]; a, bb = np.linalg.lstsq(A, hc, rcond=None)[0]
        r2 = 1 - ((hc - A @ [a, bb]) ** 2).sum() / max(((hc - hc.mean()) ** 2).sum(), 1e-12)
        out[(cb, cn, pb, pn)] = (a, bb, hc.mean(), r2)
    return out


def rand_parent(pb, pn, avoid):
    if pb == 0: return int(rng.integers(H[0]))
    r = RATE[pb]; c = np.where((r > r[pn] / 2) & (r < r[pn] * 2) & (np.arange(H[pb]) != pn))[0]
    return int(rng.choice(c)) if len(c) else int(rng.integers(H[pb]))


def boot(d, n=2000):
    r = np.random.default_rng(1); mm = np.array([d[r.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return [float(d.mean()), float(np.quantile(mm, .025)), float(np.quantile(mm, .975))]


LP0 = evaluate(W0, B0)
kl = lambda lp: (np.exp(LP0) * (LP0 - lp)).sum(1)


def run(name, fam_pairs):
    rnd = [(cb, cn, pb, rand_parent(pb, pn, None)) for cb, cn, pb, pn in fam_pairs]
    F = fits(fam_pairs + rnd); F = {k: v[:3] for k, v in F.items()}
    r2 = [fits_r2[k] for k in []] if False else None
    res = {}
    for arm, pr, mode in (("cut", fam_pairs, "cut"), ("fold_family", fam_pairs, "fold"), ("fold_random", rnd, "fold")):
        res[arm] = kl(evaluate(*apply(pr, F, mode)))
    cut = res["cut"].mean(); out = {"n_removed": len(fam_pairs), "kl": {k: float(v.mean()) for k, v in res.items()}}
    for k in ("fold_family", "fold_random"):
        out[f"share_{k}"] = boot((res["cut"] - res[k]) / cut)
    print(name, json.dumps(out), f"{time.time() - t0:.0f}s", flush=True)
    return out


results = {}
if EXP == "A":
    KEY = {"1_351": (0, 245), "3_623": (2, 61), "4_499": (1, 365), "5_135": (4, 212), "8_4": (6, 205), "10_1316": (9, 686), "11_1031": (10, 1172),
           "12_136": (9, 1137), "14_1326": (13, 78), "14_33": (12, 548), "14_831": (3, 92), "15_16": (14, 1138), "16_384": (15, 1162), "16_26": (15, 2142),
           "16_2062": (13, 1136), "17_2283": (16, 2886)}
    k6 = json.load(open(os.path.join(RES, "vision-s6-key.json"))); p6 = json.load(open(os.path.join(ROOT, "research/clean/map/s6/picks.json")))
    for k, p in p6.items(): KEY[k] = tuple(k6[k][p[0]]["neuron"])
    k8 = json.load(open(os.path.join(RES, "vision-s8-key.json")))
    for f in glob.glob(os.path.join(ROOT, "research/clean/map/s8/picks_*.json")):
        for k, p in json.load(open(f)).items():
            if p: KEY[k] = tuple(k8[k][p[0]]["neuron"])
    pairs = [tuple(map(int, k.split("_"))) + tuple(v) for k, v in KEY.items()]
    R2 = fits(pairs); results["r2_family"] = {f"{a}_{b}": round(v[3], 3) for (a, b, c, d), v in R2.items()}
    print("R^2 of child from its parent (median):", np.median([v[3] for v in R2.values()]), flush=True)
    results["A_judged"] = run("A judged parents (39 at once)", pairs)
    mp_pairs = [(cb, cn) + tuple(MAP[f"{cb}_{cn}"]["parent"]) for cb, cn, _, _ in pairs]
    results["A_map"] = run("A map v2 parents (same 39)", mp_pairs)
elif EXP == "B":
    for blk in (16,):
        en = None
        for frac in (.5, .75):
            # energy order from a small calibration sample
            d = collect([(blk, c, blk - 1, 0) for c in range(H[blk])])
            cs, _, Hc, _ = d[blk]; en = np.linalg.norm(Hc, axis=0) * np.linalg.norm(W0[blk], axis=0)
            rem = np.argsort(en)[:int(round(frac * H[blk]))]
            pairs = [(blk, int(c)) + tuple(MAP[f"{blk}_{int(c)}"]["parent"]) for c in rem]
            results[f"B_{blk}_{frac}"] = run(f"B block {blk} remove {frac:.0%}", pairs)
elif EXP == "C":
    pairs = []
    for blk in range(12, 18):
        d = collect([(blk, c, blk - 1, 0) for c in range(H[blk])]); cs, _, Hc, _ = d[blk]
        en = np.linalg.norm(Hc, axis=0) * np.linalg.norm(W0[blk], axis=0); rem = np.argsort(en)[:H[blk] // 2]
        pairs += [(blk, int(c)) + tuple(MAP[f"{blk}_{int(c)}"]["parent"]) for c in rem]
    results["C_12to17_50pct"] = run("C blocks 12-17, 50% each, composable", pairs)
json.dump(results, open(os.path.join(RES, f"vision-t1-fold-{EXP}.json"), "w"), indent=1)
