"""T3: fold variants. Rules: VISION-LOG.md T3."""
import sys
sys.argv = ["t1_fold.py", "none"]
exec(open("t1_fold.py").read().split("results = {}")[0])
MAP1 = json.load(open(os.path.join(ROOT, "research/clean/map/network_map.json")))["neurons"]
EDG = np.load(os.path.join(HF, "imagenet", "e1_edges.npz")); OFF = np.concatenate([[0], np.cumsum(H)])
loc = lambda j: (int(np.searchsorted(OFF, j, side="right") - 1), int(j - OFF[np.searchsorted(OFF, j, side="right") - 1]))

def ancestor_same_stage(cb, cn):
    p = tuple(MAP[f"{cb}_{cn}"]["parent"])
    while ST[p[0]] != ST[cb] and f"{p[0]}_{p[1]}" in MAP: p = tuple(MAP[f"{p[0]}_{p[1]}"]["parent"])
    return p if ST[p[0]] == ST[cb] else None

def family(cb, cn):
    c = [tuple(p[:2]) for p in MAP1[f"{cb}_{cn}"]["parents"]] + [tuple(MAP[f"{cb}_{cn}"]["parent"])]
    e = np.abs(EDG[str(cb)][cn]); c += [loc(int(j)) for j in np.argsort(-e)[:6]]
    out = []
    for p in c:
        if ST[p[0]] == ST[cb] and p not in out: out.append(p)
    return out[:6]

def multi_fit(children):
    """children: list of (cb, cn, [parents]). LS of h_c on its parents' hidden. Returns {(cb,cn): (coefs, b, mean, parents)}."""
    pairs = [(cb, cn, pb, pn) for cb, cn, ps in children for pb, pn in ps] + [(cb, cn, cb - 1, 0) for cb, cn, ps in children if not ps]
    data = collect(pairs); out = {}
    for cb, cn, ps in children:
        cs, plist, Hc, Hp = data[cb]; hc = Hc[:, cs.index(cn)]
        if not ps: out[(cb, cn)] = (np.zeros(0), hc.mean(), hc.mean(), []); continue
        A = np.c_[np.stack([Hp[:, plist.index(p)] for p in ps], 1), np.ones(len(hc))]
        G = A.T @ A + np.eye(A.shape[1]) * 1e-3 * np.trace(A.T @ A) / A.shape[1]; s = np.linalg.solve(G, A.T @ hc)
        out[(cb, cn)] = (s[:-1], s[-1], hc.mean(), ps)
    return out

def apply_multi(F, mode):
    W = [w.copy() for w in W0]; Bv = [b.copy() for b in B0]
    for (cb, cn) in sorted(F, key=lambda q: -q[0]):
        co, bb, mu, ps = F[(cb, cn)]; wc = W[cb][:, cn].copy(); W[cb][:, cn] = 0
        if mode == "cut" or not ps: Bv[cb] += mu * wc; continue
        Bv[cb] += bb * wc
        for a, (pb, pn) in zip(co, ps): W[pb][:, pn] += a * wc
    return W, Bv

def removal(blocks, frac):
    rem = []
    for blk in blocks:
        d = collect([(blk, c, blk - 1, 0) for c in range(H[blk])]); cs, _, Hc, _ = d[blk]
        en = np.linalg.norm(Hc, axis=0) * np.linalg.norm(W0[blk], axis=0); rem += [(blk, int(c)) for c in np.argsort(en)[:int(round(frac * H[blk]))]]
    return rem

res = {}
for name, blocks in (("block16_50", [16]), ("C_12to17_50", list(range(12, 18)))):
    rem = removal(blocks, .5); arms = {}
    arms["map_parent"] = multi_fit([(cb, cn, [tuple(MAP[f"{cb}_{cn}"]["parent"])] if ST[MAP[f"{cb}_{cn}"]["parent"][0]] == ST[cb] else []) for cb, cn in rem])
    arms["ancestor_same_stage"] = multi_fit([(cb, cn, [a] if (a := ancestor_same_stage(cb, cn)) else []) for cb, cn in rem])
    arms["family_k"] = multi_fit([(cb, cn, family(cb, cn)) for cb, cn in rem])
    KL = {"cut": kl(evaluate(*apply_multi(arms["map_parent"], "cut")))}
    for a, F in arms.items(): KL[a] = kl(evaluate(*apply_multi(F, "fold")))
    if name == "block16_50":   # dense_raw ceiling: all block-15 neurons
        cs = [cn for _, cn in rem]; pairs = [(16, cn, 15, p) for cn in cs[:1] for p in range(H[15])]
        d = collect([(16, cn, 15, 0) for cn in cs] + [(16, cs[0], 15, p) for p in range(H[15])]); cl, pl, Hc, Hp = d[16]
        Hc = Hc[:, [cl.index(c) for c in cs]]; Hp = Hp[:, [pl.index((15, p)) for p in range(H[15])]]
        R = Hc @ W0[16][:, cs].T; Cp = np.linalg.solve(Hp.T @ Hp + np.eye(H[15]) * 1e-2 * np.trace(Hp.T @ Hp) / H[15], Hp.T @ (R - R.mean(0)))
        W = [w.copy() for w in W0]; Bv = [b.copy() for b in B0]; W[16][:, cs] = 0; Bv[16] += R.mean(0) - (Hp @ Cp).mean(0); W[15] += Cp.T
        KL["dense_raw"] = kl(evaluate(W, Bv))
    cut = KL["cut"].mean(); res[name] = {"kl": {k: float(v.mean()) for k, v in KL.items()}, "share": {k: boot((KL["cut"] - v) / cut) for k, v in KL.items() if k != "cut"},
                                         "n_with_parents": {a: int(sum(1 for v in F.values() if v[3])) for a, F in arms.items()}, "n_removed": len(rem)}
    print(name, json.dumps(res[name]), f"{time.time() - t0:.0f}s", flush=True)
json.dump(res, open(os.path.join(RES, "vision-t3-variants.json"), "w"), indent=1)
