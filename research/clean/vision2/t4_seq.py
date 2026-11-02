"""T4: sequential composable family fold. Rules: VISION-LOG.md T4."""
import sys
sys.argv = ["t1_fold.py", "none"]
exec(open("t1_fold.py").read().split("results = {}")[0])
exec(open("t3_variants.py").read().split("res = {}")[0].split('exec(open("t1_fold.py")')[1].split("\n", 1)[1])

def set_weights(W, Bv):
    for b in range(NB):
        f = nn.Linear(H[b], W0[b].shape[0]); f.weight = mx.array((W[b] / G[b][:, None]).astype(np.float32)); f.bias = mx.array((Bv[b] / G[b]).astype(np.float32)); m.blocks[b].fc2 = f

def restore():
    for b in range(NB): m.blocks[b].fc2 = F2[b]

rem = removal(range(12, 18), .5); removed = set(rem)
W = [w.copy() for w in W0]; Bv = [b.copy() for b in B0]; nfold = 0
for blk in range(12, 18):
    kids = [(cb, cn) for cb, cn in rem if cb == blk]
    ch = [(cb, cn, [p for p in family(cb, cn) if p not in removed]) for cb, cn in kids]
    set_weights(W, Bv); F = multi_fit(ch); restore()          # fit on the currently edited model
    for (cb, cn), (co, bb, mu, ps) in F.items():
        wc = W[cb][:, cn].copy(); W[cb][:, cn] = 0
        if not ps: Bv[cb] += mu * wc; continue
        Bv[cb] += bb * wc; nfold += 1
        for a, (pb, pn) in zip(co, ps): W[pb][:, pn] += a * wc
    print("block", blk, "done", f"{time.time() - t0:.0f}s", flush=True)
Wc = [w.copy() for w in W0]; Bc = [b.copy() for b in B0]
cutF = multi_fit([(cb, cn, []) for cb, cn in rem])
for (cb, cn), (co, bb, mu, ps) in cutF.items(): Bc[cb] += mu * Wc[cb][:, cn]; Wc[cb][:, cn] = 0
kc = kl(evaluate(Wc, Bc)); ks = kl(evaluate(W, Bv))
out = {"n_removed": len(rem), "n_folded": nfold, "kl_cut": float(kc.mean()), "kl_seq": float(ks.mean()), "share_seq": boot((kc - ks) / kc.mean())}
print(json.dumps(out)); json.dump(out, open(os.path.join(RES, "vision-t4-seq.json"), "w"), indent=1)
