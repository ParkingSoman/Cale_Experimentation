"""T5 diagnostic: each block 12-17 alone, 50% removed: cut KL and family-k share."""
import sys
sys.argv = ["t1_fold.py", "none"]
exec(open("t1_fold.py").read().split("results = {}")[0])
exec(open("t3_variants.py").read().split("res = {}")[0].split('exec(open("t1_fold.py")')[1].split("\n", 1)[1])
out = {}
for blk in range(12, 18):
    rem = removal([blk], .5); F = multi_fit([(cb, cn, family(cb, cn)) for cb, cn in rem])
    kc = kl(evaluate(*apply_multi(F, "cut"))); kf = kl(evaluate(*apply_multi(F, "fold")))
    out[blk] = {"kl_cut": float(kc.mean()), "kl_family": float(kf.mean()), "share": boot((kc - kf) / kc.mean()), "n_with_family": int(sum(1 for v in F.values() if v[3]))}
    print(blk, json.dumps(out[blk]), flush=True)
json.dump(out, open(os.path.join(RES, "vision-t5-perblock.json"), "w"), indent=1)
