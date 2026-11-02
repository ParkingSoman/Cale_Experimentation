"""E1: can ONE neuron = parent + child? Rules: VISION-LOG.md E1."""
import json, os, time
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
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
for blk, n in ((16, 30), (13, 30), (8, 30)):
    c = [k for k in range(H[blk]) if MAP[f"{blk}_{k}"]["parent"][0] == blk - 1]
    for k in rng.choice(c, n, replace=False): pairs.append((blk, int(k)) + tuple(MAP[f"{blk}_{int(k)}"]["parent"]))
need = {}
for cb, cn, pb, pn in pairs: need.setdefault(pb, {"p": set(), "c": set()}); need[pb]["p"].add(pn); need[pb]["c"].add(cn)
# collect, per parent block pb: parent-block input (pre), z of needed parents, h of needed children (block pb+1), at the same positions
D = {pb: {"pre": [], "zp": [], "hc": [], "img": []} for pb in need}
for i in range(0, len(CAL), 50):
    x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(X[CAL[i:i + 50]])))))
    for b in range(18):
        if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
        blk = m.blocks[b]; pre = blk.pre(x)
        if b in need:
            Bn, R, C, Cd = pre.shape; sel = np.concatenate([j * R * C + rng.choice(R * C, 49, replace=False) for j in range(Bn)])
            D[b]["_sel"] = sel; D[b]["pre"].append(np.array(pre.reshape(-1, Cd)[mx.array(sel)])); ps = sorted(need[b]["p"])
            D[b]["zp"].append(np.array(blk.fc1(pre).reshape(-1, H[b])[mx.array(sel)][:, mx.array(ps)])); D[b]["img"].append(np.repeat(np.arange(i, i + Bn), 49))
        if b - 1 in need:
            cs = sorted(need[b - 1]["c"]); D[b - 1]["hc"].append(np.array(nn.gelu(blk.fc1(pre)).reshape(-1, H[b])[mx.array(D[b - 1]["_sel"])][:, mx.array(cs)]))
        x = blk(x)
for pb in D:
    for k in ("pre", "zp", "hc", "img"): D[pb][k] = np.concatenate(D[pb][k]).astype(np.float32)
print("collected", round(time.time() - t0), "s", flush=True)
gelu = lambda z: z * 0.5 * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))


def fitgd(params, loss, steps=400, lr=.02):
    opt = optim.Adam(learning_rate=lr); lg = nn.value_and_grad if False else None
    vg = mx.value_and_grad(loss)
    for _ in range(steps):
        l, g = vg(params); params = {k: params[k] - 0 for k in params}
        opt.update(params, g) if False else None
        params = {k: params[k] - lr * g[k] / (mx.sqrt(mx.mean(g[k] ** 2)) + 1e-8) * 0 for k in params} if False else params
        upd = opt.apply_gradients(g, params); params = upd
    return params


res = []
for cb, cn, pb, pn in pairs:
    d = D[pb]; ps = sorted(need[pb]["p"]); cs = sorted(need[pb]["c"])
    sub = np.random.default_rng(cn).choice(len(d["pre"]), min(4000, len(d["pre"])), replace=False)
    pre = d["pre"][sub]; zp = d["zp"][sub, ps.index(pn)]; hc = d["hc"][sub, cs.index(cn)]; hp = gelu(zp)
    tr = d["img"][sub] % 2 == 0; te = ~tr
    wp = np.array(m.blocks[pb].fc2.weight)[:, pn] * np.array(m.blocks[pb].gamma); wc = np.array(m.blocks[cb].fc2.weight)[:, cn] * np.array(m.blocks[cb].gamma)
    T = hp[:, None] * wp + hc[:, None] * wc
    def r2(P, mask): E = T[mask] - P[mask]; V = T[mask] - T[mask].mean(0); return float(1 - (E ** 2).sum() / (V ** 2).sum())
    out = {"pair": [cb, cn, pb, pn], "hand": (cb, cn, pb, pn) in HAND}
    out["F1_parent_only"] = r2(hp[:, None] * wp + hc[tr].mean() * wc, te)
    A = np.c_[hp, np.ones(len(hp))]; a, b = np.linalg.lstsq(A[tr], hc[tr], rcond=None)[0]
    out["F2_linear"] = r2(hp[:, None] * wp + (a * hp + b)[:, None] * wc, te)
    # child's pre-activation in the parent's frame: LS from parent-block input
    zc = np.log(np.maximum(hc, 1e-6))  # placeholder, replaced below
    Ap = np.c_[pre, np.ones(len(pre))]; G = Ap[tr].T @ Ap[tr]; G += np.eye(len(G)) * 1e-3 * np.trace(G) / len(G)
    # target for the child's "votes": invert GELU is ill-posed; fit z directly to the child's hidden via LS on GELU>0 rows
    u = np.linalg.solve(G, Ap[tr].T @ hc[tr]); zc_hat = Ap @ u
    out["zc_hat_r2"] = float(1 - ((hc[te] - zc_hat[te]) ** 2).sum() / ((hc[te] - hc[te].mean()) ** 2).sum())
    best = (-9, None)
    for lam in (-2, -1, -.5, 0, .5, 1, 2, 4, 8, 16):
        for cc in (-2, -1, -.5, 0, .5):
            h = gelu(zp + lam * zc_hat + cc)
            # T ~ h*(nu wp + mu wc) + bias: fit nu, mu by LS on the centered flattened target
            hcen = h - h[tr].mean(); Tc = T - T[tr].mean(0)
            Fm = np.stack([np.outer(hcen[tr], wp).ravel(), np.outer(hcen[tr], wc).ravel()], 1)
            nu, mu = np.linalg.lstsq(Fm, Tc[tr].ravel(), rcond=None)[0]
            pred = np.outer(hcen, nu * wp + mu * wc) + T[tr].mean(0)
            sc = r2(pred, tr)
            if sc > best[0]: best = (sc, (lam, cc, nu, mu), pred)
    out["F3_vote_merge"] = r2(best[2], te); out["F3_params"] = dict(zip(("lam", "c", "nu", "mu"), map(float, best[1])))
    # F0 free one neuron: read r (from parent-block input), bias, write w
    Tm, trm = mx.array(T), mx.array(tr.astype(np.float32))
    prem = mx.array(pre); r0 = np.array(m.blocks[pb].fc1.weight)[pn]; b0 = float(np.array(m.blocks[pb].fc1.bias)[pn])
    def L0(P):
        h = nn.gelu(prem @ P["r"] + P["b"]); pred = h[:, None] * P["w"][None] + P["bias"][None]
        return mx.sum(((pred - Tm) ** 2).sum(1) * trm) / mx.sum(trm)
    P = {"r": mx.array(r0), "b": mx.array(b0), "w": mx.array(wp), "bias": mx.array(hc[tr].mean() * wc)}
    opt = optim.Adam(learning_rate=.001); vg = mx.value_and_grad(L0)
    for _ in range(300):
        l, g = vg(P); opt.update(P, g); mx.eval(P)
    pred = np.array(nn.gelu(prem @ P["r"] + P["b"])[:, None] * P["w"][None] + P["bias"][None])
    out["F0_free"] = r2(pred, te)
    res.append(out)
    if out["hand"]: print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if k != "F3_params"}), out["F3_params"], flush=True)
summ = {}
for name, sel in (("hand", lambda o: o["hand"]), ("block16", lambda o: not o["hand"] and o["pair"][0] == 16), ("block13", lambda o: not o["hand"] and o["pair"][0] == 13), ("block8", lambda o: not o["hand"] and o["pair"][0] == 8)):
    rs = [o for o in res if sel(o)]
    rec = lambda o, k: (o[k] - o["F1_parent_only"]) / max(1 - o["F1_parent_only"], 1e-9)
    summ[name] = {**{k: float(np.median([o[k] for o in rs])) for k in ("F1_parent_only", "zc_hat_r2")}, **{"recover_" + k: float(np.median([rec(o, k) for o in rs])) for k in ("F2_linear", "F3_vote_merge", "F0_free")}}
    print(name, len(rs), json.dumps({k: round(v, 3) for k, v in summ[name].items()}), flush=True)
json.dump({"summary": summ, "pairs": res}, open(os.path.join(RES, "vision-e1-small.json"), "w"), indent=1)
print("done", round(time.time() - t0), "s")
