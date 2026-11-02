"""T7: composable fold with adjacent kept families. Rules: VISION-LOG.md T7."""
import sys
sys.argv = ["t1_fold.py", "none"]
exec(open("t4_seq.py").read().split("rem = removal(range(12, 18), .5)")[0].split('sys.argv = ["t1_fold.py", "none"]', 1)[1])
rem = removal(range(12, 18), .5); removed = set(rem)
def fam_adj_kept(cb, cn):
    if ST[cb - 1] != ST[cb]: return []
    f = [p for p in family(cb, cn) if p[0] == cb - 1 and p not in removed]
    if f: return f
    e = np.abs(EDG[str(cb)][cn][OFF[cb - 1]:OFF[cb]]); return [(cb - 1, int(j)) for j in np.argsort(-e) if (cb - 1, int(j)) not in removed][:3]
W = [w.copy() for w in W0]; Bv = [b.copy() for b in B0]; nfold = 0
for blk in range(12, 18):
    ch = [(cb, cn, fam_adj_kept(cb, cn)) for cb, cn in rem if cb == blk]
    set_weights(W, Bv); F = multi_fit(ch); restore()
    for (cb, cn), (co, bb, mu, ps) in F.items():
        wc = W[cb][:, cn].copy(); W[cb][:, cn] = 0
        if not ps: Bv[cb] += mu * wc; continue
        Bv[cb] += bb * wc; nfold += 1
        for a, (pb, pn) in zip(co, ps): W[pb][:, pn] += a * wc
Wc = [w.copy() for w in W0]; Bc = [b.copy() for b in B0]
for (cb, cn), (co, bb, mu, ps) in multi_fit([(cb, cn, []) for cb, cn in rem]).items(): Bc[cb] += mu * Wc[cb][:, cn]; Wc[cb][:, cn] = 0
kc = kl(evaluate(Wc, Bc)); ks = kl(evaluate(W, Bv))
out = {"n_removed": len(rem), "n_folded": nfold, "kl_cut": float(kc.mean()), "kl_fold": float(ks.mean()), "share": boot((kc - ks) / kc.mean())}
print(json.dumps(out)); json.dump(out, open(os.path.join(RES, "vision-t7-compose.json"), "w"), indent=1)
