"""A4: tree order + family fold. Rules: VISION-LOG.md A4."""
import sys
sys.argv = ["t1_fold.py", "none"]
exec(open("t1_fold.py").read().split("results = {}")[0])
exec(open("t3_variants.py").read().split("res = {}")[0].split('exec(open("t1_fold.py")')[1].split("\n", 1)[1])

def set_weights(W, Bv):
    for b in range(NB):
        f = nn.Linear(H[b], W0[b].shape[0]); f.weight = mx.array((W[b] / G[b][:, None]).astype(np.float32)); f.bias = mx.array((Bv[b] / G[b]).astype(np.float32)); m.blocks[b].fc2 = f

def restore():
    for b in range(NB): m.blocks[b].fc2 = F2[b]


TK = {}
for k, v in MAP.items(): b, n = map(int, k.split("_")); TK.setdefault(tuple(v["parent"]), []).append((b, n))
BL = list(range(12, 18)); en = {}
for blk in BL:
    d = collect([(blk, c, blk - 1, 0) for c in range(H[blk])]); cs, _, Hc, _ = d[blk]; e = np.linalg.norm(Hc, axis=0) * np.linalg.norm(W0[blk], axis=0)
    for c in range(H[blk]): en[(blk, c)] = e[c]
allN = [(b, n) for b in BL for n in range(H[b])]; yE = y[EV]
ORD = {"tree": sorted(allN, key=lambda k: (len(TK.get(k, [])), en[k])), "energy": sorted(allN, key=lambda k: en[k])}
def fam_adj_kept(cb, cn):
    if ST[cb - 1] != ST[cb]: return []
    f = [p for p in family(cb, cn) if p[0] == cb - 1 and p not in removed]
    if f: return f
    e = np.abs(EDG[str(cb)][cn][OFF[cb - 1]:OFF[cb]]); return [(cb - 1, int(j)) for j in np.argsort(-e) if (cb - 1, int(j)) not in removed][:3]

res = {}
for oname, order in ORD.items():
    for frac in (.5, .75):
        rem = order[:int(frac * len(order))]; removed = set(rem)
        Wc = [w.copy() for w in W0]; Bc = [b.copy() for b in B0]
        for (cb, cn), (co, bb, mu, ps) in multi_fit([(cb, cn, []) for cb, cn in rem]).items(): Bc[cb] += mu * Wc[cb][:, cn]; Wc[cb][:, cn] = 0
        W = [w.copy() for w in W0]; Bv = [b.copy() for b in B0]
        for blk in BL:
            ch = [(cb, cn, fam_adj_kept(cb, cn)) for cb, cn in rem if cb == blk]
            set_weights(W, Bv); F = multi_fit(ch); restore()
            for (cb, cn), (co, bb, mu, ps) in F.items():
                wc = W[cb][:, cn].copy(); W[cb][:, cn] = 0
                if not ps: Bv[cb] += mu * wc; continue
                Bv[cb] += bb * wc
                for a, (pb, pn) in zip(co, ps): W[pb][:, pn] += a * wc
        for arm, (Wx, Bx) in (("cut", (Wc, Bc)), ("fold", (W, Bv))):
            lp = evaluate(Wx, Bx); res[f"{oname}_{frac}_{arm}"] = {"top1": float((lp.argmax(1) == yE).mean()), "kl": float(kl(lp).mean())}
            print(oname, frac, arm, res[f"{oname}_{frac}_{arm}"], f"{time.time() - t0:.0f}s", flush=True)
json.dump(res, open(os.path.join(RES, "vision-a4-dropdown.json"), "w"))
