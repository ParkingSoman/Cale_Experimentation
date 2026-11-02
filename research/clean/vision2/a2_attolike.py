"""A2: leaves-first collapse vs a real small model. Rules: VISION-LOG.md A2."""
import sys
sys.argv = ["t1_fold.py", "none"]
exec(open("t1_fold.py").read().split("results = {}")[0])
ma = load_convnext("atto", (2, 2, 6, 2), (40, 80, 160, 320)); LA = logits(ma, np.asarray(X[EV])); pa = LA.argmax(1); del ma
yE = y[EV]; p0 = LP0.argmax(1); D = (p0 == yE) & (pa != yE)
kids = {}
for k, v in MAP.items():
    b, n = map(int, k.split("_")); kids.setdefault(tuple(v["parent"]), []).append((b, n))
desc = {}
for b in range(17, -1, -1):
    for n in range(H[b]): desc[(b, n)] = sum(1 + desc[c] for c in kids.get((b, n), []))
BL = list(range(12, 18)); en = {}; mu = {}
for blk in BL:
    d = collect([(blk, c, blk - 1, 0) for c in range(H[blk])]); cs, _, Hc, _ = d[blk]
    e = np.linalg.norm(Hc, axis=0) * np.linalg.norm(W0[blk], axis=0)
    for c in range(H[blk]): en[(blk, c)] = e[c]; mu[(blk, c)] = Hc[:, c].mean()
allN = [(b, n) for b in BL for n in range(H[b])]; print("desc 0 share", np.mean([desc[k] == 0 for k in allN]), flush=True)
ORD = {"tree": sorted(allN, key=lambda k: (desc[k], en[k])), "energy": sorted(allN, key=lambda k: en[k]),
       "random": [allN[i] for i in np.random.default_rng(5).permutation(len(allN))]}
res = {}
for name, order in ORD.items():
    for frac in (.25, .5, .75):
        W = [w.copy() for w in W0]; Bv = [b.copy() for b in B0]
        for (b, n) in order[:int(frac * len(order))]: Bv[b] += mu[(b, n)] * W[b][:, n]; W[b][:, n] = 0
        lp = evaluate(W, Bv); p = lp.argmax(1)
        res[f"{name}_{frac}"] = {"top1": float((p == yE).mean()), "agree_atto": float((p == pa).mean()), "attolike_err": float((p[D] == pa[D]).mean()),
                                 "kl_from_tiny": float(kl(lp).mean()), "per_img_attolike": (p[D] == pa[D]).astype(int).tolist(), "per_img_correct": (p == yE).astype(int).tolist()}
        print(name, frac, {k: round(v, 4) for k, v in res[f"{name}_{frac}"].items() if not k.startswith("per")}, f"{time.time() - t0:.0f}s", flush=True)
res["base"] = {"tiny_top1": float((p0 == yE).mean()), "atto_top1": float((pa == yE).mean()), "tiny_agree_atto": float((p0 == pa).mean()), "nD": int(D.sum())}
print(res["base"]); json.dump(res, open(os.path.join(RES, "vision-a2-attolike.json"), "w"))
