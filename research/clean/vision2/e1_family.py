"""E1b-d: concentration, family check of the top-edge parent vs a breadth-matched null, gallery. Rules: VISION-LOG E1."""
import json
import numpy as np
from convnext import *
E = np.load(os.path.join(HF, "imagenet", "e1_edges.npz")); M = np.load(os.path.join(HF, "imagenet", "m3_matrices.npz"))
H = [384]*3 + [768]*3 + [1536]*9 + [3072]*3; OFF = np.concatenate([[0], np.cumsum(H)]); blk = np.concatenate([[b]*H[b] for b in range(18)])
rng = np.random.default_rng(11); out = {}; pairs = []
print("block | top share of + input | top parent: broader / contains>=.8 / lift med | null: broader / contains / lift | dist med | same as M3 best")
for b in range(1, 18):
    e = E[str(b)]; pos = np.maximum(e, 0); top = e.argmax(1); share = pos.max(1) / np.maximum(pos.sum(1), 1e-12)
    rate = M[f"{b}_rate"]; Cp = M[f"{b}_point_pos_C"].astype(np.float32); Cv = M[f"{b}_view_pos_C"].astype(np.float32)
    bp = M[f"{b}_point_pos_br"]; bv = M[f"{b}_view_pos_br"]
    def stats(par):
        n = np.arange(len(par)); cp, cv = Cp[n, par], Cv[n, par]; usev = cv / bv[par] > cp / bp[par]
        c = np.where(usev, cv, cp); br = np.where(usev, bv[par], bp[par])
        return br > rate, c >= .8, c / br
    bro, con, lift = stats(top)
    null = np.array([rng.choice(np.where((blk == blk[t]) & (np.abs(bp - bp[t]) <= .2 * bp[t]) & (np.arange(len(bp)) != t))[0]) if ((blk == blk[t]) & (np.abs(bp - bp[t]) <= .2 * bp[t])).sum() > 1 else t for t in top])
    nbro, ncon, nlift = stats(null)
    # M3 best containment parent (positive pole, either variant, narrowest with C>=.8 and broader, cap .5)
    okp = (Cp >= .8) & (bp[None] > rate[:, None]) & (bp[None] <= .5); okv = (Cv >= .8) & (bv[None] > rate[:, None]) & (bv[None] <= .5)
    same = float(np.mean([(okp[i, top[i]] or okv[i, top[i]]) for i in range(len(top))]))
    dist = b - blk[top]
    o = {"top_share_med": float(np.median(share)), "broader": float(bro.mean()), "contains": float(con.mean()), "lift_med": float(np.median(lift)),
         "null_broader": float(nbro.mean()), "null_contains": float(ncon.mean()), "null_lift_med": float(np.median(nlift)),
         "dist_med": float(np.median(dist)), "dist_hist": {int(k): float(v) for k, v in zip(*np.unique(dist, return_counts=True))},
         "is_an_M3_parent": same, "family_pass": float((bro & con).mean())}
    out[b] = o
    print(f"{b:5d} | {o['top_share_med']:.3f} | {o['broader']:.2f} / {o['contains']:.2f} / {o['lift_med']:.1f} | {o['null_broader']:.2f} / {o['null_contains']:.2f} / {o['null_lift_med']:.1f} | {o['dist_med']:.0f} | {same:.2f}")
    if b >= 3 and b % 3 == 0 or b == 17:
        i = int(rng.choice(len(top))); pairs.append((b, i, int(blk[top[i]]), int(top[i] - OFF[blk[top[i]]]), float(share[i]), float(lift[i]), bool(con[i])))
w = np.array(H[1:]); tot = lambda k: float(sum(out[b][k] * H[b] for b in out) / w.sum())
print(f"whole network: top parent broader {tot('broader'):.3f}, contains {tot('contains'):.3f}, both {tot('family_pass'):.3f}; null contains {tot('null_contains'):.3f}")
out["whole"] = {k: tot(k) for k in ("broader", "contains", "family_pass", "null_contains")}
json.dump(out, open(os.path.join(RES, "vision-e1-family.json"), "w"), indent=1)
from gallery import gallery
rows = []
for b, n, pbk, pn, sh, lf, c in pairs:
    rows += [(b, n, "child"), (pbk, pn, f"   ^ its strongest input (share {sh:.2f} of + input), lift {lf:.1f}, contains: {c}")]
print(gallery(rows, "edge_pairs.png", "Child, then the earlier neuron that feeds it most (direct connection)"))
