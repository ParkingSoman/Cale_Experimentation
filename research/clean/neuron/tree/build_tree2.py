"""N5 step 1b: capability trees. Roots = the R most concentrated, broad neurons of one stage-3 block. Members = neurons of
any later block that are group-specific (concentration > CONC, breadth > BR) AND drop when the roots are filled with
their goal means, above a noise floor (99th percentile of the same candidates' drops when 5 random sets of R root-block
neurons are filled instead). Bar: remove the members (goal-mean fills): KL on group test photos against other test photos,
and the same for random same-count sets per block (3 draws). Saves results/tree/trees2.json.
"""
import json
import os
import sys

import numpy as np
import torch

from build_tree import photos, root_scores
from common import DEV, OUT, Fills, Peaks, blocks, coarse_kl_rows, groups, kl_rows, load, net, split

torch.set_grad_enabled(False)
R, CONC, BR = 8, .4, .25


def main():
    which = sys.argv[1:] or ["dogs", "birds", "vehicles"]
    prof = dict(np.load(os.path.join(OUT, "profile_P10.npz")))
    G = groups()
    gn = list(prof["groups"])
    ph = net.Photos("read")
    fit = split(ph.lab)
    m = load("base")
    BL = blocks(m)
    H = [blk.block[3].out_features for blk in BL]
    fills, pk = Fills(m), Peaks(m)
    rng = np.random.default_rng(5)
    res = {}
    for name in which:
        grp = G[name]
        gi = gn.index(name)
        gmean = lambda g, ids: prof[f"gm{g}"][gi][np.asarray(ids)]
        best = None
        for g in range(15, 25):
            conc, br = root_scores(prof, g, grp)
            sc = conc * br
            top = np.argsort(-sc)[:R]
            if best is None or sc[top].sum() > best[0]:
                best = (float(sc[top].sum()), g, top)
        _, l, roots = best
        x = ph.tensor(photos(ph, grp, fit, 10, max(1, -(-120 // len(grp))), 120)).to(DEV)

        def run(d):
            fills.set(d)
            st = {}
            for a in range(0, len(x), 60):
                m(x[a:a + 60])
                for g, v in pk.store.items():
                    st.setdefault(g, []).append(v.clone())
            fills.clear()
            return {g: torch.cat(v) for g, v in st.items()}
        b0 = run({})

        def drops(ids):
            b1 = run({l: (ids, gmean(l, ids))})
            return {g: ((b0[g] - b1[g]) / torch.from_numpy(prof[f"sd{g}"]).to(DEV)).mean(0).cpu().numpy()
                    for g in range(l + 1, 30)}
        dr = drops(roots)
        pool = np.setdiff1d(np.arange(H[l]), roots)
        nz = [drops(rng.choice(pool, R, replace=False)) for _ in range(5)]
        levels = []
        for g in range(l + 1, 30):
            conc, br = root_scores(prof, g, grp)
            cand = np.where((conc > CONC) & (br > BR))[0]
            if len(cand) == 0:
                continue
            floor = float(np.percentile(np.concatenate([n[g][cand] for n in nz]), 99))
            mem = cand[dr[g][cand] > floor]
            levels.append(dict(block=g, ids=mem.tolist(), n_candidates=int(len(cand)), floor=floor,
                               drops=dr[g][mem].round(3).tolist(), block_size=H[g]))
            print(f"[{name}] block {g}: candidates {len(cand)}, members {len(mem)}, floor {floor:.3f}", flush=True)
        levels = [v for v in levels if v["ids"]]
        te_g = photos(ph, grp, ~fit, 0, max(1, -(-120 // len(grp))), 120)
        others = np.setdiff1d(np.where(~fit)[0], np.where(np.isin(ph.lab, grp))[0])
        te_o = np.sort(np.random.default_rng(3).choice(others, 300, replace=False))

        def score(d):
            out = {}
            for tag, ids in (("group", te_g), ("other", te_o)):
                kl, ckl = [], []
                for a in range(0, len(ids), 60):
                    xx = ph.tensor(ids[a:a + 60]).to(DEV)
                    yr = m(xx)
                    fills.set(d)
                    yy = m(xx)
                    fills.clear()
                    kl.append(kl_rows(yr, yy).cpu())
                    ckl.append(coarse_kl_rows(yr, yy, grp).cpu())
                out[tag] = (torch.cat(kl).numpy(), torch.cat(ckl).numpy())
            return out
        tf = {v["block"]: (v["ids"], gmean(v["block"], v["ids"])) for v in levels}
        st = score(tf)
        rs = []
        for _ in range(3):
            rf = {}
            for g, (ids, _) in tf.items():
                conc, br = root_scores(prof, g, grp)
                pool = np.where(~((conc > CONC) & (br > BR)))[0]
                q = rng.choice(pool, len(ids), replace=False)
                rf[g] = (q, gmean(g, q))
            rs.append(score(rf))
        bar = dict(tree_group=float(st["group"][0].mean()), tree_other=float(st["other"][0].mean()),
                   tree_group_coarse=float(st["group"][1].mean()),
                   random_group=float(np.mean([r["group"][0].mean() for r in rs])),
                   random_other=float(np.mean([r["other"][0].mean() for r in rs])))
        bar["tree_ratio_group_over_other"] = bar["tree_group"] / max(bar["tree_other"], 1e-9)
        bar["random_ratio_group_over_other"] = bar["random_group"] / max(bar["random_other"], 1e-9)
        n = sum(len(v["ids"]) for v in levels)
        print(f"[{name}] roots block {l}, members {n} in {len(levels)} blocks; bar {json.dumps(bar)}", flush=True)
        res[name] = dict(root_block=l, roots=roots.tolist(), levels=levels, n_members=n, bar=bar,
                         test_group=te_g.tolist(), test_other=te_o.tolist())
        json.dump(res, open(os.path.join(OUT, "trees2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
