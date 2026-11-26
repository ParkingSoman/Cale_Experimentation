"""N5 step 1b: find each capability's root and its tree, then test whether the tree carries the capability.

Root: among stage-3 blocks 15-24, the neuron with the largest  concentration x breadth  on the group, from the class
profile (profile_P10.npz):  z_c = (class mean peak - median over classes) / std over classes;
concentration = sum_group relu(z) / sum_all relu(z);  breadth = share of group classes with z > 1.
Tree: level 0 = {root}. Level j = neurons of block l+j whose per-photo peak drops (in std units of the profile) when
level j-1 is filled with its goal-photo means, on tree photos (group fit photos not used by the profile), above a noise
floor: 99.9th percentile of the same scores when 5 random same-size sets of block l+j-1 are filled instead. Cap 64.
Bar: remove the whole tree (goal-mean fills); KL to Base on group test photos and on other test photos, against
3 draws of the same number of random non-tree neurons per block (paired bootstrap interval).
"""
import json
import os
import sys
import time

import numpy as np
import torch

from common import DEV, OUT, Fills, Peaks, blocks, coarse_kl_rows, groups, kl_rows, load, net, split

torch.set_grad_enabled(False)
CAP = 64


def root_scores(prof, g, grp):
    cm = prof[f"cm{g}"].astype(np.float32)
    z = (cm - np.median(cm, 0)) / (cm.std(0) + 1e-6)
    rz = np.maximum(z, 0)
    conc = rz[grp].sum(0) / (rz.sum(0) + 1e-9)
    br = (z[grp] > 1).mean(0)
    return conc, br


def in_group_share(prof, g, ids, grp):
    conc, _ = root_scores(prof, g, grp)
    return conc[ids]


def photos(ph, grp, fit, first, per, cap):
    out = []
    for c in grp:
        w = np.where((ph.lab == c) & fit)[0][first:first + per]
        out += w.tolist()
    rng = np.random.default_rng(0)
    out = np.array(out)
    if len(out) > cap:
        out = rng.choice(out, cap, replace=False)
    return np.sort(out)


def main():
    which = sys.argv[1:] or ["dogs", "birds", "vehicles"]
    prof = dict(np.load(os.path.join(OUT, "profile_P10.npz")))
    G = groups()
    gnames = list(prof["groups"])
    ph = net.Photos("read")
    fit = split(ph.lab)
    m = load("base")
    BL = blocks(m)
    H = [blk.block[3].out_features for blk in BL]
    fills = Fills(m)
    pk = Peaks(m)
    rng = np.random.default_rng(1)

    def run(x, d):
        fills.set(d)
        ys, st = [], {}
        for a in range(0, len(x), 60):
            ys.append(m(x[a:a + 60]))
            for g, v in pk.store.items():
                st.setdefault(g, []).append(v.clone())
        fills.clear()
        return torch.cat(ys), {g: torch.cat(v) for g, v in st.items()}

    res = {}
    for name in which:
        grp = G[name]
        gi = gnames.index(name)
        gmean = lambda g, ids: prof[f"gm{g}"][gi][ids]
        # root
        best = None
        cands = []
        for g in range(15, 25):
            conc, br = root_scores(prof, g, grp)
            sc = conc * br
            for i in np.argsort(-sc)[:3]:
                cands.append((float(sc[i]), g, int(i), float(conc[i]), float(br[i])))
        cands.sort(reverse=True)
        _, l, r, rc, rb = cands[0]
        print(f"[{name}] root block {l} neuron {r}: concentration {rc:.2f}, breadth {rb:.2f}", flush=True)
        tp = photos(ph, grp, fit, 10, max(1, -(-120 // len(grp))), 120)
        x = ph.tensor(tp).to(DEV)
        y0, base = run(x, {})
        levels = [dict(block=l, ids=[r], score=[None])]
        t0 = time.time()
        for j in range(1, 30 - l):
            gp, par = l + j - 1, levels[-1]["ids"]
            gc = l + j
            sd = torch.from_numpy(prof[f"sd{gc}"]).to(DEV)

            def drops(ids):
                _, pkf = run(x, {gp: (ids, gmean(gp, ids))})
                return ((base[gc] - pkf[gc]) / sd).mean(0).cpu().numpy()
            s = drops(par)
            noise = []
            pool = np.setdiff1d(np.arange(H[gp]), par)
            for _ in range(5):
                noise.append(drops(rng.choice(pool, len(par), replace=False)))
            floor = float(np.percentile(np.concatenate(noise), 99.9))
            kids = np.where(s > floor)[0]
            kids = kids[np.argsort(-s[kids])][:CAP]
            share = in_group_share(prof, gc, kids, grp) if len(kids) else np.array([])
            lev = dict(block=gc, ids=kids.tolist(), score=s[kids].round(3).tolist(), floor=floor,
                       n_above_floor=int((s > floor).sum()), block_size=H[gc],
                       in_group_majority=float((share > 0.5).mean()) if len(kids) else None,
                       in_group_median=float(np.median(share)) if len(kids) else None)
            print(f"  level {j} (block {gc}): {lev['n_above_floor']} above floor {floor:.3f}, kept {len(kids)}, "
                  f"in-group majority {lev['in_group_majority']}, median share {lev['in_group_median']} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            if len(kids) == 0:
                break
            levels.append(lev)
        # bar: remove the whole tree vs random same-count sets
        te_g = photos(ph, grp, ~fit, 0, max(1, -(-120 // len(grp))), 120)
        others = np.setdiff1d(np.where(~fit)[0], np.where(np.isin(ph.lab, grp))[0])
        te_o = np.sort(np.random.default_rng(3).choice(others, 300, replace=False))
        tree_fill = {}
        for lev in levels[1:]:
            tree_fill[lev["block"]] = (lev["ids"], gmean(lev["block"], lev["ids"]))

        def score(d):
            out = {}
            for tag, ids in (("group", te_g), ("other", te_o)):
                kl, ckl = [], []
                for a in range(0, len(ids), 60):
                    xx = ph.tensor(ids[a:a + 60]).to(DEV)
                    fills.clear()
                    yr = m(xx)
                    fills.set(d)
                    yy = m(xx)
                    fills.clear()
                    kl.append(kl_rows(yr, yy).cpu())
                    ckl.append(coarse_kl_rows(yr, yy, grp).cpu())
                out[tag] = (torch.cat(kl).numpy(), torch.cat(ckl).numpy())
            return out
        st = score(tree_fill)
        rnd = []
        for d in range(3):
            rf = {}
            for g, (ids, _) in tree_fill.items():
                pool = np.setdiff1d(np.arange(H[g]), ids)
                q = rng.choice(pool, len(ids), replace=False)
                rf[g] = (q, gmean(g, q))
            rnd.append(score(rf))
        bar = {}
        for tag in ("group", "other"):
            a = st[tag][0]
            b = np.mean([r[tag][0] for r in rnd], 0)
            diff = a - b
            bs = [diff[np.random.default_rng(k).integers(0, len(diff), len(diff))].mean() for k in range(2000)]
            bar[tag] = dict(tree_kl=float(a.mean()), random_kl=float(b.mean()),
                            tree_coarse_kl=float(st[tag][1].mean()),
                            random_coarse_kl=float(np.mean([r[tag][1].mean() for r in rnd])),
                            diff_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        print(f"  bar: {json.dumps(bar)}", flush=True)
        res[name] = dict(root=dict(block=l, neuron=r, concentration=rc, breadth=rb, candidates=cands[:8]),
                         tree_photos=len(tp), levels=levels, n_members=sum(len(v["ids"]) for v in levels[1:]),
                         bar=bar)
        json.dump(res, open(os.path.join(OUT, "trees.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
