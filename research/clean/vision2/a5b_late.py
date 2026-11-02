"""A5: whole-network dropdown slider. Rules: VISION-LOG.md A5."""
import sys
sys.argv = ["t1_fold.py", "none"]
exec(open("t1_fold.py").read().split("results = {}")[0])
yE = y[EV]
kids = {}
for k, v in MAP.items():
    b, n = map(int, k.split("_")); kids.setdefault(tuple(v["parent"]), []).append((b, n))
en = {}; mu = {}
for blk in range(12, NB):
    d = collect([(blk, c, max(blk - 1, 0), 0) for c in range(H[blk])]); cs, _, Hc, _ = d[blk]
    e = np.linalg.norm(Hc, axis=0) * np.linalg.norm(W0[blk], axis=0)
    for c in range(H[blk]): en[(blk, c)] = e[c]; mu[(blk, c)] = Hc[:, c].mean()
print("energy done", f"{time.time() - t0:.0f}s", flush=True)
rr = np.random.default_rng(5)
BLS = range(12, NB)
ORD = {"tree": {b: sorted(range(H[b]), key=lambda n: (len(kids.get((b, n), [])), en[(b, n)])) for b in BLS},
       "energy": {b: sorted(range(H[b]), key=lambda n: en[(b, n)]) for b in BLS},
       "random": {b: list(rr.permutation(H[b])) for b in []}}
res = {}
for frac in (.5, .75):
    for name, O in [(k, v) for k, v in ORD.items() if v]:
        W = [w.copy() for w in W0]; Bv = [b.copy() for b in B0]
        for b in O:
            for n in O[b][:int(frac * H[b])]: Bv[b] += mu[(b, n)] * W[b][:, n]; W[b][:, n] = 0
        lp = evaluate(W, Bv); res[f"{name}_{frac}"] = {"top1": float((lp.argmax(1) == yE).mean()), "kl": float(kl(lp).mean())}
        print(name, frac, res[f"{name}_{frac}"], f"{time.time() - t0:.0f}s", flush=True)
json.dump(res, open(os.path.join(RES, "vision-a5b-late-equal.json"), "w"))
