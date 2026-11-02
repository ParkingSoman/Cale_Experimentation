"""E3: merge vs fold with readers refit. Rules: VISION-LOG.md E3."""
import sys
sys.argv = ["t1_fold.py", "none"]
exec(open("t1_fold.py").read().split("results = {}")[0])
exec(open("t3_variants.py").read().split("res = {}")[0].split('exec(open("t1_fold.py")')[1].split("\n", 1)[1])
gelu = lambda z: z * 0.5 * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))
rem = removal([16], .5); F = multi_fit([(cb, cn, family(cb, cn)) for cb, cn in rem])
d = collect([(16, cn, 15, 0) for _, cn in rem]); cs, _, Hc, _ = d[16]; en = {cn: np.linalg.norm(Hc[:, cs.index(cn)]) * np.linalg.norm(W0[16][:, cn]) for cn in cs}
par = {}
for _, cn in rem:
    p = tuple(MAP[f"16_{cn}"]["parent"])
    if p[0] == 15 and (p not in par or en[cn] > en[par[p]]): par[p] = cn
print("parents to merge:", len(par), flush=True)
# data: block-15 input (pre), z of parents, z of chosen children, same positions
Pre, Zp, Zc, IMG = [], [], [], []; pl = sorted(par); cl = [par[p] for p in pl]
for i in range(0, len(CAL), 50):
    x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(X[CAL[i:i + 50]])))))
    for b in range(15): 
        if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
        x = m.blocks[b](x)
    x = m.down[2](m.down_norm[2](x)); pre = m.blocks[15].pre(x); x2 = m.blocks[15](x); pre16 = m.blocks[16].pre(x2)
    Bn = pre.shape[0]; Pre.append(np.array(pre.reshape(-1, pre.shape[-1]))); Zp.append(np.array(m.blocks[15].fc1(pre).reshape(-1, H[15])[:, mx.array([p[1] for p in pl])]))
    Zc.append(np.array(m.blocks[16].fc1(pre16).reshape(-1, H[16])[:, mx.array(cl)])); IMG.append(np.repeat(np.arange(i, i + Bn), 49))
Pre, Zp, Zc, IMG = [np.concatenate(a).astype(np.float64) for a in (Pre, Zp, Zc, IMG)]; tr = IMG % 2 == 0
P1 = np.c_[Pre, np.ones(len(Pre))]; Gm = P1[tr].T @ P1[tr]; Gm += np.eye(len(Gm)) * 1e-3 * np.trace(Gm) / len(Gm)
U = np.linalg.solve(Gm, P1[tr].T @ Zc[tr])                                   # (C+1, nparents): child votes in the parent frame
W1 = np.array(m.blocks[15].fc1.weight, np.float64); b1 = np.array(m.blocks[15].fc1.bias, np.float64)

def build(arm):
    Wm, Bm = [w.copy() for w in W0], [b.copy() for b in B0]; W1n, b1n = W1.copy(), b1.copy(); merged = set(); OM = 5
    if arm == "merge":
        for j, (p, cn) in enumerate(zip(pl, cl)):
            zp, zc = Zp[:, j], Zc[:, j]; zch = P1 @ U[:, j]; hp, hc = gelu(zp), gelu(zc); wp, wc = W0[15][:, p[1]], W0[16][:, cn]
            gpp, gpc, gcc = wp @ wp, wp @ wc, wc @ wc; best = (np.inf, None)
            for lam in (0, .25, .5, 1, 2, 4):
                for cc in (-1, -.5, 0, .5, 1):
                    hs = gelu(zp + lam * zch + cc); hcn = hs - hs.mean(); hpn = hp - hp.mean(); hcc = hc - hc.mean()
                    A = np.array([[OM * gpp, gpc], [gpc, gcc]]) * (hcn ** 2).sum(); bv = np.array([OM * gpp * (hcn * hpn).sum() + gpc * (hcn * hcc).sum(), gpc * (hcn * hpn).sum() + gcc * (hcn * hcc).sum()])
                    nu, mu = np.linalg.solve(A + 1e-12 * np.eye(2), bv); u_, v_ = hpn - nu * hcn, hcc - mu * hcn
                    e = (OM * u_ * u_ * gpp + 2 * u_ * v_ * gpc + v_ * v_ * gcc).sum()
                    if e < best[0]: best = (e, (lam, cc, nu, mu, hs.mean(), hp.mean(), hc.mean()))
            lam, cc, nu, mu, hsm, hpm, hcm = best[1]; PRM[(p, cn)] = best[1]
            W1n[p[1]] += lam * U[:-1, j]; b1n[p[1]] += lam * U[-1, j] + cc
            Wm[15][:, p[1]] = nu * wp + mu * wc; Bm[15] += hpm * wp + hcm * wc - hsm * (nu * wp + mu * wc)
            Wm[16][:, cn] = 0; merged.add(cn)
    for (cb, cn), (co, bb, mu_, ps) in F.items():
        if cn in merged: continue
        wc = Wm[cb][:, cn].copy(); Wm[cb][:, cn] = 0
        if arm == "cut" or not ps: Bm[cb] += mu_ * wc; continue
        Bm[cb] += bb * wc
        for a, (pb, pn) in zip(co, ps): Wm[pb][:, pn] += a * wc
    return Wm, Bm, (W1n, b1n) if arm == "merge" else None

def lin(Wt, bt):
    f = nn.Linear(Wt.shape[1], Wt.shape[0]); f.weight = mx.array(Wt.astype(np.float32)); f.bias = mx.array(bt.astype(np.float32)); return f

def install(Wm, Bm, fc1s):
    for b in range(NB):
        f = nn.Linear(H[b], W0[b].shape[0]); f.weight = mx.array((Wm[b] / G[b][:, None]).astype(np.float32)); f.bias = mx.array((Bm[b] / G[b]).astype(np.float32)); m.blocks[b].fc2 = f
    for b, f in fc1s.items(): m.blocks[b].fc1 = f

def uninstall():
    for b in range(NB): m.blocks[b].fc2 = F2[b]; m.blocks[b].fc1 = F1[b]

def pres(blk):   # pre-activation input of block blk on CAL images, current weights
    out = []
    for i in range(0, len(CAL), 50):
        x = m.stream(mx.array(norm_img(np.asarray(X[CAL[i:i + 50]])))) if False else m.stem_norm(m.stem(mx.array(norm_img(np.asarray(X[CAL[i:i + 50]])))))
        for b in range(blk):
            if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
            x = m.blocks[b](x)
        if ST[blk] != ST[blk - 1]: x = m.down[ST[blk] - 1](m.down_norm[ST[blk] - 1](x))
        p = m.blocks[blk].pre(x); out.append(np.array(p.reshape(-1, p.shape[-1])))
    return np.concatenate(out).astype(np.float64)

F1 = [m.blocks[b].fc1 for b in range(NB)]; PRM = {}
ORIG = {b: pres(b) for b in (16, 17)}
ZO = {b: ORIG[b] @ np.array(F1[b].weight, np.float64).T + np.array(F1[b].bias, np.float64) for b in (16, 17)}
res = {}
for arm in ("cut", "fold", "merge"):
    Wm, Bm, f15 = build(arm); fc1s = {15: lin(*f15)} if f15 else {}
    install(Wm, Bm, fc1s); lp = logp(); res[arm] = kl(lp)
    for b in ((16, 17) if f15 else (17,)):
        P = pres(b); P1r = np.c_[P, np.ones(len(P))]; Gr = P1r.T @ P1r; Gr += np.eye(len(Gr)) * 1e-3 * np.trace(Gr) / len(Gr)
        S = np.linalg.solve(Gr, P1r.T @ ZO[b]); fc1s[b] = lin(S[:-1].T, S[-1]); install(Wm, Bm, fc1s)
    res[arm + "+refit"] = kl(logp()); uninstall()
    print(arm, res[arm].mean(), res[arm + "+refit"].mean(), f"{time.time() - t0:.0f}s", flush=True)
# refit alone on the intact model (should be ~0)
fc1s = {}
P = pres(17); P1r = np.c_[P, np.ones(len(P))]; Gr = P1r.T @ P1r; Gr += np.eye(len(Gr)) * 1e-3 * np.trace(Gr) / len(Gr)
S = np.linalg.solve(Gr, P1r.T @ ZO[17]); fc1s[17] = lin(S[:-1].T, S[-1]); install(W0, B0, fc1s); res["intact+refit"] = kl(logp()); uninstall()
kc = res["cut"].mean()
out = {"kl": {k: float(v.mean()) for k, v in res.items()}, "share_vs_cut": {k: boot((res["cut"] - v) / kc) for k, v in res.items() if k != "cut"},
       "merge_minus_fold_refit": boot((res["fold+refit"] - res["merge+refit"]) / kc), "params": {f"{p}_{c}": v[:4] for (p, c), v in PRM.items()}}
print(json.dumps({k: out[k] for k in ("kl", "share_vs_cut", "merge_minus_fold_refit")}))
json.dump(out, open(os.path.join(RES, "vision-e3-refit.json"), "w"), indent=1, default=float)
