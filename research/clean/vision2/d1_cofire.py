"""D1g: are the tiny parts of one small neuron a co-firing group in TINY alone? For each small neuron in the consensus json:
mean pairwise corr (GELU outputs) among its top-10 parts vs 10 random alive tiny neurons from the same blocks; and the
rank of each part among the first part's most co-firing tiny neurons (could tiny alone have found the group?).
usage: python d1_cofire.py SMALL STAGE"""
import os, sys, json
import numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "d1_consensus.py")).read().split("Zs, ids_s = collect")[0].replace("small, stage, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])", "small, stage, N = sys.argv[1], int(sys.argv[2]), 0"))
Zs, ids_s = collect(small, [stage]); Zt, ids_t = collect("tiny", [stage - 1, stage])
Ht = gl(Zt); alive = (Zt > 0).mean(0) > .005; Hz = (Ht - Ht.mean(0)) / (Ht.std(0) + 1e-9); del Ht
rows = json.load(open(os.path.join(HF, "imagenet", "d1", f"consensus_{small}_s{stage}.json"))); pos = {tuple(v): i for i, v in enumerate(ids_s)}
out = []
for r in rows:
    j = pos[tuple(int(v) for v in r["id"])]; t = gl(Zs[:, j].astype(np.float64)); tz = (t - t.mean()) / t.std()
    c = (Hz * tz[:, None].astype(np.float32)).mean(0); P = np.argsort(-c)[:10]
    C = (Hz[:, P].T @ Hz[:, P]) / len(Hz); pc = C[np.triu_indices(10, 1)].mean()
    blocks = [ids_t[p][0] for p in P]; G_ = np.random.default_rng(len(out)); rnd = [G_.choice(np.where(alive & (np.array([b for b, _ in ids_t]) == b))[0]) for b in blocks]
    Cr = (Hz[:, rnd].T @ Hz[:, rnd]) / len(Hz); rc = Cr[np.triu_indices(10, 1)].mean()
    co = (Hz * Hz[:, P[0]][:, None]).mean(0); co[P[0]] = -9; rk = np.argsort(np.argsort(-co))[P[1:]]
    out.append((r["id"], r["best_part_corr"], pc, rc, int(np.median(rk))))
print(f"{small} stage {stage}: parts' mean pairwise corr vs random same-block; median co-firing rank of parts 2-10 around part 1 (of {len(ids_t)})")
for o in sorted(out, key=lambda o: -o[1])[:15]: print(f"  {str(tuple(o[0])):10s} best {o[1]:.2f}  parts {o[2]:.3f}  random {o[3]:.3f}  median rank {o[4]}")
A = np.array([o[2:] for o in out], float); print("medians: parts %.3f random %.3f rank %.0f" % tuple(np.median(A, 0)))
