"""E1 fast: one neuron = parent + child? All errors in closed form (a pair writes along 2 fixed directions)."""
import json, os, time
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
t0 = time.time(); m = load_convnext(); X, y = load_images(); ST = m.stage_of
H = [m.blocks[b].fc1.weight.shape[0] for b in range(18)]
rank = np.zeros(len(y), int)
for c in range(1000):
    ix = np.where(y == c)[0]; rank[ix] = np.arange(len(ix))
CAL = np.where(rank < 1)[0]; rng = np.random.default_rng(0)
MAP = json.load(open(os.path.join(ROOT, "research/clean/map/network_map_v2.json")))["neurons"]
HAND = [(5, 135, 4, 212), (10, 1316, 9, 686), (11, 1031, 10, 1172), (14, 1326, 13, 78), (16, 384, 15, 1162), (16, 26, 15, 2142), (17, 2283, 16, 2886)]
pairs = list(HAND)
for blk in (16, 13, 8):
    c = [k for k in range(H[blk]) if MAP[f"{blk}_{k}"]["parent"][0] == blk - 1]
    for k in rng.choice(c, 100, replace=False): pairs.append((blk, int(k)) + tuple(MAP[f"{blk}_{int(k)}"]["parent"]))
need = {}
for cb, cn, pb, pn in pairs: need.setdefault(pb, {"p": set(), "c": set()}); need[pb]["p"].add(pn); need[pb]["c"].add(cn)
need = {b: {k: sorted(v) for k, v in d.items()} for b, d in need.items()}
D = {pb: {"pre": [], "zp": [], "zc": [], "img": []} for pb in need}
for i in range(0, len(CAL), 50):
    x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(X[CAL[i:i + 50]]))))); sel = {}
    for b in range(18):
        if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
        blk = m.blocks[b]; pre = blk.pre(x)
        if b in need:
            Bn, R, C, Cd = pre.shape; sel[b] = mx.array(np.concatenate([j * R * C + rng.choice(R * C, 49, replace=False) for j in range(Bn)]))
            D[b]["pre"].append(np.array(pre.reshape(-1, Cd)[sel[b]])); D[b]["zp"].append(np.array(blk.fc1(pre).reshape(-1, H[b])[sel[b]][:, mx.array(need[b]["p"])]))
            D[b]["img"].append(np.repeat(np.arange(i, i + Bn), 49))
        if b - 1 in need: D[b - 1]["zc"].append(np.array(blk.fc1(pre).reshape(-1, H[b])[sel[b - 1]][:, mx.array(need[b - 1]["c"])]))
        x = blk(x)
for pb in D:
    for k in D[pb]: D[pb][k] = np.concatenate(D[pb][k]).astype(np.float64)
print("collected", round(time.time() - t0), "s", flush=True)
gelu = lambda z: z * 0.5 * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))
res = []
for cb, cn, pb, pn in pairs:
    d = D[pb]; zp = d["zp"][:, need[pb]["p"].index(pn)]; zc = d["zc"][:, need[pb]["c"].index(cn)]; hp, hc = gelu(zp), gelu(zc)
    tr = d["img"] % 2 == 0; te = ~tr
    wp = np.array(m.blocks[pb].fc2.weight, np.float64)[:, pn] * np.array(m.blocks[pb].gamma); wc = np.array(m.blocks[cb].fc2.weight, np.float64)[:, cn] * np.array(m.blocks[cb].gamma)
    gpp, gpc, gcc = wp @ wp, wp @ wc, wc @ wc
    def err(al, be, mask):     # ||(hp-al) wp + (hc-be) wc||^2 summed
        u, v = hp[mask] - al[mask], hc[mask] - be[mask]; return (u * u * gpp + 2 * u * v * gpc + v * v * gcc).sum()
    tot = lambda mask: err(np.full(len(hp), hp[tr].mean()), np.full(len(hc), hc[tr].mean()), mask)
    def fit2(hs):              # pred = hs*(nu wp + mu wc) + const: LS for nu, mu in the wp/wc metric (train), errors on test
        hcn = hs - hs[tr].mean(); hpn = hp - hp[tr].mean(); hcc = hc - hc[tr].mean()
        A = np.array([[gpp, gpc], [gpc, gcc]]) * (hcn[tr] ** 2).sum()
        bvec = np.array([gpp * (hcn * hpn)[tr].sum() + gpc * (hcn * hcc)[tr].sum(), gpc * (hcn * hpn)[tr].sum() + gcc * (hcn * hcc)[tr].sum()])
        nu, mu = np.linalg.solve(A + 1e-12 * np.eye(2), bvec)
        return err(hp[tr].mean() + nu * hcn, hc[tr].mean() + mu * hcn, te), (nu, mu)
    T0 = tot(te); o = {"pair": [cb, cn, pb, pn], "hand": (cb, cn, pb, pn) in HAND}
    e1 = err(hp, np.full(len(hc), hc[tr].mean()), te)                       # F1 parent only (child -> mean)
    A2 = np.c_[hp, np.ones(len(hp))]; a, b = np.linalg.lstsq(A2[tr], hc[tr], rcond=None)[0]; e2 = err(hp, a * hp + b, te)   # F2 linear fold
    P = np.c_[d["pre"], np.ones(len(hp))]; G = P[tr].T @ P[tr]; G += np.eye(len(G)) * 1e-3 * np.trace(G) / len(G)
    zch = P @ np.linalg.solve(G, P[tr].T @ zc[tr])                          # child's pre-activation from the parent block's input
    o["zc_hat_r2"] = float(1 - ((zc - zch)[te] ** 2).sum() / ((zc[te] - zc[te].mean()) ** 2).sum())
    best3, bestO = (np.inf, None), (np.inf, None)
    for lam in (-1, -.5, -.25, 0, .25, .5, 1, 2, 4):
        for cc in (-2, -1, -.5, 0, .5, 1):
            e, prm = fit2(gelu(zp + lam * zch + cc))
            if e < best3[0]: best3 = (e, (lam, cc) + tuple(prm))
            e, prm = fit2(gelu(zp + lam * zc + cc))
            if e < bestO[0]: bestO = (e, (lam, cc) + tuple(prm))
    rec = lambda e: float((e1 - e) / max(e1, 1e-12))
    o.update({"F1_r2": float(1 - e1 / T0), "rec_F2_linear": rec(e2), "rec_F3_vote_merge": rec(best3[0]), "rec_F3_oracle_true_zc": rec(bestO[0]),
              "F3_params": best3[1], "oracle_params": bestO[1]})
    res.append(o)
    if o["hand"]: print(o["pair"], {k: round(v, 3) for k, v in o.items() if k.startswith("rec") or k in ("F1_r2", "zc_hat_r2")}, "lam,c,nu,mu", np.round(best3[1], 2), flush=True)
summ = {}
for name, sel in (("hand", lambda o: o["hand"]), ("block16", lambda o: not o["hand"] and o["pair"][0] == 16), ("block13", lambda o: not o["hand"] and o["pair"][0] == 13), ("block8", lambda o: not o["hand"] and o["pair"][0] == 8)):
    rs = [o for o in res if sel(o)]
    summ[name] = {k: float(np.median([o[k] for o in rs])) for k in ("F1_r2", "zc_hat_r2", "rec_F2_linear", "rec_F3_vote_merge", "rec_F3_oracle_true_zc")}
    print(name, len(rs), json.dumps({k: round(v, 3) for k, v in summ[name].items()}), flush=True)
json.dump({"summary": summ, "pairs": res}, open(os.path.join(RES, "vision-e1-small.json"), "w"), indent=1, default=float)
print("done", round(time.time() - t0), "s")
