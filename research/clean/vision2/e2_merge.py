"""E2: vote merge in real weights, block 16. Rules: VISION-LOG.md E2."""
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
Wm, Bm = [w.copy() for w in W0], [b.copy() for b in B0]; W1n, b1n = W1.copy(), b1.copy(); merged = set()
for j, (p, cn) in enumerate(zip(pl, cl)):
    zp, zc = Zp[:, j], Zc[:, j]; zch = P1 @ U[:, j]; hp, hc = gelu(zp), gelu(zc); wp, wc = W0[15][:, p[1]], W0[16][:, cn]
    gpp, gpc, gcc = wp @ wp, wp @ wc, wc @ wc; best = (np.inf, None)
    for lam in (0, .25, .5, 1, 2, 4):
        for cc in (-1, -.5, 0, .5, 1):
            hs = gelu(zp + lam * zch + cc); hcn = hs - hs.mean(); hpn = hp - hp.mean(); hcc = hc - hc.mean()
            A = np.array([[gpp, gpc], [gpc, gcc]]) * (hcn ** 2).sum(); bv = np.array([gpp * (hcn * hpn).sum() + gpc * (hcn * hcc).sum(), gpc * (hcn * hpn).sum() + gcc * (hcn * hcc).sum()])
            nu, mu = np.linalg.solve(A + 1e-12 * np.eye(2), bv); u_, v_ = hpn - nu * hcn, hcc - mu * hcn
            e = (u_ * u_ * gpp + 2 * u_ * v_ * gpc + v_ * v_ * gcc).sum()
            if e < best[0]: best = (e, (lam, cc, nu, mu, hs.mean(), hp.mean(), hc.mean()))
    lam, cc, nu, mu, hsm, hpm, hcm = best[1]
    W1n[p[1]] += lam * U[:-1, j]; b1n[p[1]] += lam * U[-1, j] + cc
    Wm[15][:, p[1]] = nu * wp + mu * wc; Bm[15] += hpm * wp + hcm * wc - hsm * (nu * wp + mu * wc)
    Wm[16][:, cn] = 0; merged.add(cn)
for (cb, cn), (co, bb, mu_, ps) in F.items():        # the rest: T3 family fold
    if cn in merged: continue
    wc = Wm[cb][:, cn].copy(); Wm[cb][:, cn] = 0
    if not ps: Bm[cb] += mu_ * wc; continue
    Bm[cb] += bb * wc
    for a, (pb, pn) in zip(co, ps): Wm[pb][:, pn] += a * wc
f1 = m.blocks[15].fc1; nf = nn.Linear(W1.shape[1], H[15]); nf.weight = mx.array(W1n.astype(np.float32)); nf.bias = mx.array(b1n.astype(np.float32))
m.blocks[15].fc1 = nf; kmerge = kl(evaluate(Wm, Bm)); m.blocks[15].fc1 = f1
kc = kl(evaluate(*apply_multi(F, "cut"))); kf = kl(evaluate(*apply_multi(F, "fold")))
out = {"n_removed": len(rem), "n_merged": len(merged), "kl": {"cut": float(kc.mean()), "family": float(kf.mean()), "merge+family": float(kmerge.mean())},
       "share_family": boot((kc - kf) / kc.mean()), "share_merge_family": boot((kc - kmerge) / kc.mean()), "merge_minus_family": boot((kf - kmerge) / kc.mean())}
print(json.dumps(out)); json.dump(out, open(os.path.join(RES, "vision-e2-merge.json"), "w"), indent=1)
