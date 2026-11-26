"""Checks on "Read, one firing at a time" (advisor review, 2026-11-23). Mac, numpy (+ networkx via oc_subcircuits).
C1 counterpart = on-rate? per subcircuit, A neurons ranked by on-rate alone vs by mean match; each counterpart's on-rate
   and its cosine when on.
C2 shared second feeder? the counterparts' own top feeders (channel votes) on the subcircuit's DEV firings where on.
C3 groups of inputs or of the read? partition agreement (adjusted Rand) of DEV groups: real vs shuffled-weight reads.
C4 best A neuron per CHECK firing, among those on: its match vs the same with shuffled reads (paired 99th pct);
   how many distinct A neurons that takes.
C5 the s3:69 vote split by stream source (skip = block-0 output: block-0 hidden + stage-2 + their shifts; block-1 write).
Usage: python oc_subcheck.py [pos]
"""
import os
import sys

import numpy as np

import oc_subcircuits as oc

R = oc.R


def ari(a, b):
    from math import comb
    ct = np.zeros((a.max() + 1, b.max() + 1), int)
    for x, y in zip(a, b):
        ct[x, y] += 1
    s = sum(comb(int(v), 2) for v in ct.ravel())
    sa = sum(comb(int(v), 2) for v in ct.sum(1))
    sb = sum(comb(int(v), 2) for v in ct.sum(0))
    e = sa * sb / comb(len(a), 2)
    mx = (sa + sb) / 2
    return (s - e) / (mx - e) if mx != e else 1.0


def main(pole):
    W = np.load(os.path.join(R, "oc_firesplit_w.npz"))
    P = dict(np.load(os.path.join(R, f"oc_firesplit_{pole}.npz")))
    C = np.load(os.path.join(R, f"oc_subcircuits_{pole}.npz"))
    dpre = np.abs(P["pre"] - P["pre_cache"]) / np.maximum(np.abs(P["pre_cache"]), 1e-6)
    keep = (P["err_h3"] <= 1e-2) & (dpre <= 1e-2)
    n0 = len(keep)
    P = {k: (v[keep] if hasattr(v, "shape") and v.ndim and len(v) == n0 else v) for k, v in P.items()}
    S = P["split"].astype(np.float64)
    dev, chk, g = C["dev"], C["chk"], C["g_dev"]
    AS, on = oc.a_splits(W, pole, P)
    U = oc.unit(S)
    cosA = np.einsum("nc,njc->nj", U, oc.unit(AS))                         # cosine whether on or not
    G = g.max() + 1

    print("C1  counterpart = on-rate?  (per subcircuit: top 3 by mean match | top 3 by on-rate alone)")
    for k in range(G):
        m = dev[g == k]
        rate = on[m].mean(0)
        mm = C["cps"][k]
        by_m, by_r = np.argsort(-mm)[:3], np.argsort(-rate)[:3]
        desc = []
        for j in by_m:
            w = on[m, j]
            desc.append(f"n{j}: on {rate[j]:.2f}, cos when on {cosA[m][w, j].mean() if w.any() else float('nan'):.2f}")
        print(f"  k{k}: by match {['n%d' % j for j in by_m]} | by on-rate {['n%d' % j for j in by_r]}; " + "; ".join(desc))

    print("\nC2  shared second feeder?  (n101's top 4 feeders vs each counterpart's top 4, mean vote on the DEV firings where on)")
    for k in range(G):
        m = dev[g == k]
        nf = np.argsort(-S[m].mean(0))[:4]
        print(f"  k{k}: n101 " + ", ".join(f"s3:{c}" for c in nf))
        for j in np.argsort(-C["cps"][k])[:3]:
            w = on[m, j]
            if not w.any():
                print(f"       n{j}: never on")
                continue
            v = AS[m][w, j].mean(0)
            print(f"       n{j} (on {w.sum()}): " + ", ".join(f"s3:{c} {v[c]:+.2f}" for c in np.argsort(-v)[:4])
                  + f"  | its vote on n101's 2nd feeder s3:{nf[1]} = {v[nf[1]]:+.2f} (rank {int((v > v[nf[1]]).sum()) + 1})")

    print("\nC3  groups of inputs or of the read?  adjusted Rand of DEV partitions")
    w9 = W["w"].reshape(256, 9) * W["sc"]
    centre = P["patch"][:, :, 2:5, 2:5].reshape(len(S), 256, 9)
    Fd = oc.feat(S[dev])
    aris, shs = [], []
    for s in range(20):
        pi = np.random.default_rng(1000 + s).permutation(256)
        Ss = np.einsum("ct,nct->nc", w9[pi], centre)
        gs = oc.groups(oc.feat(Ss[dev]))
        aris.append(ari(g, gs))
        shs.append(gs)
    pair = [ari(shs[i], shs[i + 1]) for i in range(0, 20, 2)]
    reseed = [ari(g, oc.groups(Fd, seed=s)) for s in range(1, 6)]
    print(f"  real vs shuffled read: ARI median {np.median(aris):.2f} (range {min(aris):.2f}-{max(aris):.2f})")
    print(f"  shuffled vs shuffled: ARI median {np.median(pair):.2f};  real vs real with another Louvain seed: median {np.median(reseed):.2f}")

    print("\nC4  best A neuron on at each CHECK firing (not one picked neuron)")
    Mon = np.where(on, cosA, -np.inf)
    best = Mon[chk].max(1)
    arg = Mon[chk].argmax(1)
    has = np.isfinite(best)
    sh_best = np.empty((200, len(chk)))
    for s in range(200):
        pi = np.random.default_rng(1000 + s).permutation(256)
        Ss = np.einsum("ct,nct->nc", w9[pi], centre)
        cs = np.einsum("nc,njc->nj", oc.unit(Ss[chk]), oc.unit(AS[chk]))
        sh_best[s] = np.where(on[chk], cs, -np.inf).max(1)
    q = np.quantile(sh_best, .99, 0)
    ok = has & (best > q)
    print(f"  firings with any A neuron on: {has.mean():.2f}; best-on match median {np.median(best[has]):.2f} "
          f"(shuffled 99th median {np.median(q[has]):.2f}); best beats its own 99th: {ok.mean():.2f} of CHECK")
    for lv in (.5, .7):
        print(f"  best-on match >= {lv} and beats chance: {(ok & (best >= lv)).mean():.2f}")
    u, c = np.unique(arg[ok], return_counts=True)
    order = np.argsort(-c)
    cum = np.cumsum(c[order]) / max(ok.sum(), 1)
    k80 = int(np.searchsorted(cum, .8)) + 1 if ok.any() else 0
    print(f"  distinct A neurons that are the best passing match: {len(u)}; {k80} cover 80% of the passing firings; top: "
          + ", ".join(f"n{u[i]} {c[i]}" for i in order[:10]))

    print("\nC5  the s3:69 vote split by stream source (firings where s3:69 is among the 20 largest votes)")
    top, st = P["top"], P["src_top"]
    for ch in (69, 88, 203, 86, 78):
        rows, cols = np.where(top == ch)
        if not len(rows):
            print(f"  s3:{ch}: never in the top 20")
            continue
        x = st[rows, cols]                                                  # m x 644
        b1 = x[:, :256].sum(1) + x[:, 640]
        b0h = x[:, 256:512].sum(1)
        s2 = x[:, 512:640].sum(1)
        sh0 = x[:, 641] + x[:, 642]
        tot = x.sum(1)
        skip = b0h + s2 + sh0
        h0 = np.argsort(-x[:, 256:512].mean(0))[:3]
        h1 = np.argsort(-x[:, :256].mean(0))[:3]
        print(f"  s3:{ch} ({len(rows)} firings, mean vote {tot.mean():+.3f}): skip share {np.median(skip / tot):.2f} "
              f"[block-0 hidden {np.median(b0h / tot):+.2f}, stage-2 {np.median(s2 / tot):+.2f}, shifts {np.median(sh0 / tot):+.2f}]; "
              f"block-1 write {np.median(b1 / tot):+.2f}; top block-0 hidden " + ", ".join(f"h{j}" for j in h0)
              + "; top block-1 hidden " + ", ".join(f"h{j}" for j in h1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pos")
