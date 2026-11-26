"""Pair study, steps 4 and 5 (first pass, per-photo peaks): what does the Small neuron s do beyond its counterpart p?

Data: per-photo peak pre-activations of every stage-4 hidden neuron in both models, on 3,000 fit photos (tools.py).
These 3,000 photos split in two halves: SEL (choose neurons) and EVAL (score). All R^2 are on EVAL.

For each confident pair (s, p) with p in block 0 or 1 (tree.json):
  R2[p]          s ~ p
  R2[p+kids]     + p's 6 children by virtual weight (tree.json)
  R2[p+nbr]      + 6 Base neurons that correlate with p like the children do, but are not children (VW rank > 500)
  R2[p+kids2]    + the 6 children of the second-best firing match p2
  R2[p+rand]     + 6 random Base neurons
  R2[p+resid]    + the 6 Base neurons whose peaks best correlate with s's residual after p (chosen on SEL)
For the residual neurons: are they orphans, children of p, in which block, how much do they correlate with p?

P4 (children absorbed): for each child c of p, take c's top 1% photos (SEL+EVAL). Compare s and p there, both z-scored.
If s fires more than p on c's best photos, s covers c beyond p. Split by orphan child against matched child.
"""
import json
import os

import numpy as np

from cache import OUT


def z(a):
    return (a - a.mean(0)) / (a.std(0) + 1e-6)


def r2(y, X, sel, ev):
    X1 = np.c_[np.ones(len(y)), X]
    w, *_ = np.linalg.lstsq(X1[sel], y[sel], rcond=None)
    e = y[ev] - X1[ev] @ w
    return 1 - e.var() / y[ev].var()


def main():
    pk = np.load(os.path.join(OUT, "tools_peaks_3000.npz"))
    S = z(pk["small"].astype(np.float32))  # (n, 3, 3072)
    B = z(pk["base"].astype(np.float32))   # (n, 3, 4096)
    n = S.shape[0]
    Bf = B.reshape(n, -1)                   # flat index = block * 4096 + neuron
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    sel, ev = np.zeros(n, bool), np.zeros(n, bool)
    sel[perm[: n // 2]] = True
    ev[perm[n // 2:]] = True
    tree = json.load(open(os.path.join(OUT, "tree.json")))
    orphans = {b: set() for b in range(3)}
    for b in range(3):
        Rf = np.load(os.path.join(OUT, f"tools_peakR_b{b}.npy")).astype(np.float32)
        Rw = np.load(os.path.join(OUT, f"writesR_b{b}.npy")).astype(np.float32)
        orphans[b] = set(range(4096)) - set(Rf.argmax(1).tolist()) - set(Rw.argmax(1).tolist())
    # correlations of every Base neuron with every other, on SEL (for neighbourhood controls)
    Bs = z(Bf[sel])
    from tree import base_parts, vw  # noqa: E402
    from cache import load  # noqa: E402
    P = base_parts(load("base"))
    VW = {(k, k2): vw(P, k, k2).numpy() for k in (0, 1) for k2 in range(k + 1, 3)}
    rows = []
    pairs = tree["pairs"]
    idx = rng.permutation(len(pairs))[:600]  # a random 600 of the 3,649 confident pairs
    for q in [pairs[i] for i in idx]:
        k, s_, p = q["block"], q["s"], q["p"]
        y = S[:, k, s_]
        pf = k * 4096 + p
        kids = [d["block"] * 4096 + d["c"] for d in q["kids"]]
        # VW rank of every later-block neuron as a child of p
        vw_rank = {}
        for k2 in range(k + 1, 3):
            order = np.argsort(-VW[(k, k2)][p])
            rk = np.empty(4096, int)
            rk[order] = np.arange(4096)
            for c in range(4096):
                vw_rank[k2 * 4096 + c] = rk[c]
        cp = Bs.T @ Bs[:, pf] / Bs.shape[0]  # corr of every Base neuron with p on SEL
        # neighbourhood: same blocks as kids, corr with p close to each kid's, not a child
        nbr = []
        for c in kids:
            blk = c // 4096
            cand = np.arange(blk * 4096, (blk + 1) * 4096)
            cand = cand[[vw_rank.get(i, 0) > 500 for i in cand]]
            cand = np.setdiff1d(cand, kids + nbr + [pf])
            nbr.append(int(cand[np.argmin(np.abs(cp[cand] - cp[c]))]))
        # second-best firing match and its kids
        Rf = np.load(os.path.join(OUT, f"tools_peakR_b{k}.npy")).astype(np.float32)[s_]
        p2 = int(np.argsort(-Rf)[1])
        kids2 = []
        for k2 in range(k + 1, 3):
            kids2 += [(k2 * 4096 + int(c), VW[(k, k2)][p2, c]) for c in np.argsort(-VW[(k, k2)][p2])[:6]]
        kids2 = [c for c, _ in sorted(kids2, key=lambda t: -t[1])[:6]]
        rand = rng.choice(np.setdiff1d(np.arange(Bf.shape[1]), [pf]), 6, replace=False).tolist()
        # residual neurons
        X1 = np.c_[np.ones(n), Bf[:, pf]]
        w, *_ = np.linalg.lstsq(X1[sel], y[sel], rcond=None)
        res_ = y - X1 @ w
        cr = z(Bf[sel]).T @ z(res_[sel][:, None])[:, 0] / sel.sum()
        cr[pf] = -np.inf
        resid = np.argsort(-cr)[:6].tolist()
        row = dict(block=k, s=s_, p=p, fire=q["fire"], write=q["write"],
                   r2_p=r2(y, Bf[:, [pf]], sel, ev),
                   r2_kids=r2(y, Bf[:, [pf] + kids], sel, ev),
                   r2_nbr=r2(y, Bf[:, [pf] + nbr], sel, ev),
                   r2_kids2=r2(y, Bf[:, [pf] + kids2], sel, ev),
                   r2_rand=r2(y, Bf[:, [pf] + rand], sel, ev),
                   r2_resid=r2(y, Bf[:, [pf] + resid], sel, ev),
                   resid=[dict(f=int(c), block=int(c // 4096), orphan=int(c % 4096) in orphans[c // 4096],
                               vw_rank=int(vw_rank.get(c, -1)), corr_p=float(cp[c]), corr_res=float(cr[c]))
                          for c in resid])
        # P4: s against p on each child's top 1% photos
        p4 = []
        for d in q["kids"]:
            c = d["block"] * 4096 + d["c"]
            top = np.argsort(-Bf[:, c])[: max(n // 100, 10)]
            p4.append(dict(orphan=d["orphan"], s_minus_p=float(y[top].mean() - Bf[top, pf].mean())))
        row["p4"] = p4
        rows.append(row)
    keys = ["r2_p", "r2_kids", "r2_nbr", "r2_kids2", "r2_rand", "r2_resid"]
    summ = {k_: float(np.median([r[k_] for r in rows])) for k_ in keys}
    gain = lambda a: np.array([r[a] - r["r2_p"] for r in rows])
    summ["gain_kids_minus_nbr_median"] = float(np.median(gain("r2_kids") - gain("r2_nbr")))
    summ["kids_beat_nbr_share"] = float(np.mean(gain("r2_kids") > gain("r2_nbr")))
    summ["kids_beat_kids2_share"] = float(np.mean(gain("r2_kids") > gain("r2_kids2")))
    summ["kids_beat_rand_share"] = float(np.mean(gain("r2_kids") > gain("r2_rand")))
    R = [d for r in rows for d in r["resid"]]
    summ["resid_orphan_share"] = float(np.mean([d["orphan"] for d in R]))
    summ["resid_same_block_as_p_share"] = float(np.mean([d["block"] == r["block"] for r in rows for d in r["resid"]]))
    summ["resid_later_block_vw_rank_median"] = float(np.median([d["vw_rank"] for d in R if d["vw_rank"] >= 0] or [-1]))
    summ["resid_corr_p_median"] = float(np.median([d["corr_p"] for d in R]))
    summ["orphan_share_base"] = float(np.mean([len(orphans[b]) / 4096 for b in range(3)]))
    P4o = [d["s_minus_p"] for r in rows for d in r["p4"] if d["orphan"]]
    P4m = [d["s_minus_p"] for r in rows for d in r["p4"] if not d["orphan"]]
    summ["p4_s_minus_p_orphan_kids_mean"] = float(np.mean(P4o))
    summ["p4_s_minus_p_matched_kids_mean"] = float(np.mean(P4m))
    summ["p4_share_s_above_p_orphan"] = float(np.mean(np.array(P4o) > 0))
    for k_, v in summ.items():
        print(f"{k_:40s} {v:.3f}")
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(OUT, "mix_peaks.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
