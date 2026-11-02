"""T6: family restricted to the block right before the child. Rules: VISION-LOG.md T6."""
import sys
sys.argv = ["t1_fold.py", "none"]
exec(open("t1_fold.py").read().split("results = {}")[0])
exec(open("t3_variants.py").read().split("res = {}")[0].split('exec(open("t1_fold.py")')[1].split("\n", 1)[1])
def fam_adj(cb, cn):
    f = [p for p in family(cb, cn) if p[0] == cb - 1]
    if f: return f
    e = np.abs(EDG[str(cb)][cn][OFF[cb - 1]:OFF[cb]]); return [(cb - 1, int(j)) for j in np.argsort(-e)[:3]]
out = {}
for blk in (12, 13, 14, 16, 17):
    rem = removal([blk], .5); F = multi_fit([(cb, cn, fam_adj(cb, cn)) for cb, cn in rem])
    kc = kl(evaluate(*apply_multi(F, "cut"))); kf = kl(evaluate(*apply_multi(F, "fold")))
    out[blk] = {"kl_cut": float(kc.mean()), "kl_family_adj": float(kf.mean()), "share": boot((kc - kf) / kc.mean())}
    print(blk, json.dumps(out[blk]), flush=True)
json.dump(out, open(os.path.join(RES, "vision-t6-adjacent.json"), "w"), indent=1)
