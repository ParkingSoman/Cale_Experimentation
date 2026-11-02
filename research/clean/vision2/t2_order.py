"""T2: removal order from the tree. Rules: VISION-LOG.md T2."""
import sys
sys.argv = ["t1_fold.py", "none"]
exec(open("t1_fold.py").read().split("results = {}")[0])
kids = {}
for k, v in MAP.items(): kids.setdefault(tuple(v["parent"]), []).append(tuple(map(int, k.split("_"))))
memo = {}
def desc(node):
    if node in memo: return memo[node]
    memo[node] = sum(1 + desc(c) for c in kids.get(node, [])); return memo[node]
out = {}
for blk in (14, 16):
    d = collect([(blk, c, blk - 1, 0) for c in range(H[blk])]); cs, _, Hc, _ = d[blk]
    en = np.linalg.norm(Hc, axis=0) * np.linalg.norm(W0[blk], axis=0); mu = Hc.mean(0)
    ds = np.array([desc((blk, n)) for n in range(H[blk])])
    print(f"block {blk}: neurons with no descendants {np.mean(ds == 0):.2f}, median descendants {np.median(ds)}", flush=True)
    orders = {"leaves_first": np.lexsort((en, ds)), "low_energy": np.argsort(en), "random": rng.permutation(H[blk]), "parents_first": np.lexsort((en, -ds))}
    for frac in (.25, .5, .75):
        k = int(round(frac * H[blk]))
        for name, o in orders.items():
            W = [w.copy() for w in W0]; Bv = [b.copy() for b in B0]; rem = o[:k]
            Bv[blk] += W[blk][:, rem] @ mu[rem]; W[blk][:, rem] = 0
            v = kl(evaluate(W, Bv)); out[f"{blk}_{frac}_{name}"] = boot(v)
            print(f"block {blk} remove {frac:.0%} {name:14s} KL {v.mean():.4f}", flush=True)
json.dump(out, open(os.path.join(RES, "vision-t2-order.json"), "w"), indent=1)
