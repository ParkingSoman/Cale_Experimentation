"""Read, one firing at a time (LOG "Read, one firing at a time", 2026-11-23): subcircuits, counterparts, bins.
Mac, numpy + networkx (run with research/clean/curves/.venv/bin/python). Inputs from code/oc_firesplit.py:
results/oc_firesplit_w.npz and results/oc_firesplit_<pole>.npz. No labels are loaded.

Per pole:
  DEV / CHECK = the firings split in two halves by photo (numpy seed 0).
  Subcircuits = Louvain communities (seed 0, resolution 1) on the 15-nearest-neighbour cosine graph of DEV firings;
    feature = (driver part, suppressor part) of n101's 256-channel split, unit length.
  A's split = each A neuron's per-channel vote at its best cell in the 3 x 3 neighbourhood; ON = that best input is at
    least half of the neuron's own max. Firing match = cosine of signed splits if ON, else 0.
  Counterpart = per subcircuit, the A neuron with the highest mean firing match over its DEV firings.
  Chance = 200 shuffles of n101's weight rows over the channels, same firings, whole pipeline again; paired per firing.
  Bins over CHECK firings: (a) explained, (b) named gap, (c) unexplained (definitions in the LOG).
Saved: results/oc_subcircuits_<pole>.npz. Prints the bins as a progress bar, P1, and one table per subcircuit.
Usage: python oc_subcircuits.py [pos|neg] [n_shuffles] [--smoke]
"""
import os
import sys

import networkx as nx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
K_NN, MIN_G, NSH = 15, 10, 200
# Children of s3:69 by dependence, picked without n101 (record list, not fitted).
# Derived from: LOG "Children, the essay's way" / "Checks" C2 (receiver in its own strong half, c69 / pre at the cell).
CHILDREN = {305: .51, 297: .49, 163: .48, 97: .41, 23: .40, 108: .32, 237: .31, 275: .29}
SRC_NAMES = ([f"s3b1:h{j}" for j in range(256)] + [f"s3b0:h{j}" for j in range(256)] + [f"s2:{c}" for c in range(128)]
             + ["shift s3b1.bn2", "shift s3b0.bn2", "shift ds.bn", "shift n101.bn1"])


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)


def feat(S):
    return unit(np.concatenate([np.clip(S, 0, None), np.clip(-S, 0, None)], 1))


def windows(patch):
    """patch: n x 256 x 7 x 7 -> n x 9 x 256 x 9: the 3 x 3 input window of each cell of the 3 x 3 neighbourhood."""
    out = np.empty((patch.shape[0], 9, patch.shape[1], 9), np.float32)
    for k in range(9):
        dr, dc = k // 3, k % 3
        out[:, k] = patch[:, :, 2 * dr:2 * dr + 3, 2 * dc:2 * dc + 3].reshape(patch.shape[0], patch.shape[1], 9)
    return out


def a_splits(W, pole, P):
    """A's per-channel votes at each neuron's best neighbourhood cell, and its ON mask.
    pos: best = highest input, ON = at least half the neuron's own max.
    neg: best = lowest input, ON = at most half the neuron's own min (the A neuron is pushed down there too)."""
    A9 = P["A_pre9"]                                                         # n x 512 x 9
    if pole == "neg":
        A9 = np.where(np.isfinite(A9), A9, np.inf)
        best = A9.argmin(2)
        on = A9.min(2) <= .5 * W["A_min"][None]
    else:
        best = A9.argmax(2)                                                  # n x 512
        on = A9.max(2) >= .5 * W["A_max"][None]
    win = windows(P["patch"])                                                # n x 9 x 256 x 9
    Aw = W["A_w"].reshape(512, 256, 9) * W["A_sc"][:, None, None]
    n = len(win)
    AS = np.empty((n, 512, 256), np.float32)
    for i in range(n):
        allc = np.einsum("jct,kct->jkc", Aw, win[i])                         # 512 x 9 x 256
        AS[i] = allc[np.arange(512), best[i]]
    return AS, on


def match(S, AS, on):
    return np.einsum("nc,njc->nj", unit(S), unit(AS)) * on                  # n x 512


def groups(Fd, seed=0):
    sim = Fd @ Fd.T
    np.fill_diagonal(sim, -1)
    G = nx.Graph()
    G.add_nodes_from(range(len(Fd)))
    for i in range(len(Fd)):
        for j in np.argsort(-sim[i])[:K_NN]:
            G.add_edge(i, int(j), weight=float(max(sim[i, j], 0)))
    comms = nx.community.louvain_communities(G, weight="weight", resolution=1, seed=seed)
    lab = np.empty(len(Fd), int)
    for g, cm in enumerate(sorted(comms, key=len, reverse=True)):
        lab[list(cm)] = g
    return lab


def pipeline(S, M, dev, chk, seed=0):
    """S: n x 256 split; M: n x 512 match. Returns per-CHECK-firing nearest-centroid cosine, its group, its
    counterpart match, and the per-group facts."""
    Fd, Fc = feat(S[dev]), feat(S[chk])
    g = groups(Fd, seed)
    G = g.max() + 1
    cen = unit(np.stack([Fd[g == k].mean(0) for k in range(G)]))
    cs = Fc @ cen.T
    near = cs.argmax(1)
    cos_near = cs.max(1)
    cps = np.stack([M[dev][g == k].mean(0) for k in range(G)])               # G x 512
    cp = cps.argmax(1)
    cp_match = M[chk][np.arange(len(chk)), cp[near]]
    return dict(g_dev=g, near=near, cos_near=cos_near, cp=cp, cps=cps, cp_match=cp_match, cen=cen)


def bar(frac, width=40):
    k = int(round(frac * width))
    return "#" * k + "." * (width - k)


def main(pole, nsh, smoke):
    W = np.load(os.path.join(R, "oc_firesplit_w.npz"))
    P = dict(np.load(os.path.join(R, f"oc_firesplit_{pole}{'_smoke' if smoke else ''}.npz")))
    if "pre_cache" in P:
        # LOG deviation 2026-11-23: a firing whose photo run misses the cache (stage-3 output or n101's input, relative
        # error above 1e-2) is dropped from its pole. Decided before any analysis was read.
        dpre = np.abs(P["pre"] - P["pre_cache"]) / np.maximum(np.abs(P["pre_cache"]), 1e-6)
        keep = (P["err_h3"] <= 1e-2) & (dpre <= 1e-2)
        n0 = len(keep)
        P = {k: (v[keep] if hasattr(v, "shape") and v.ndim and len(v) == n0 else v) for k, v in P.items()}
        print(f"cache check: dropped {n0 - keep.sum()} of {n0} firings")
    S = P["split"].astype(np.float64)
    n = len(S)
    print(f"pole {pole}: {n} firings; exact-sum err max {P['err_sum'].max():.1e}, top-20 err max {P['err_top'].max():.1e}, "
          f"h3 vs cache err max {P['err_h3'].max():.1e}")
    if P["err_sum"].max() > 1e-3 or P["err_top"].max() > 1e-3:
        print("EXACT SUMS FAIL: nothing else is read (LOG bar).")
        return
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    dev, chk = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
    AS, on = a_splits(W, pole, P)
    recon = AS.sum(2) + W["A_sh"][None]
    A9 = P["A_pre9"]
    ref = np.where(np.isfinite(A9), A9, np.inf).min(2) if pole == "neg" else A9.max(2)
    a_err = np.abs(recon - ref).max()
    print(f"A splits recomputed from patch vs A input at the best cell: max abs diff {a_err:.2e}; "
          f"A neurons ON per firing, median {int(np.median(on.sum(1)))}")
    if a_err > 1e-2:
        print("A SPLITS DO NOT ADD UP: stop (patch or cell geometry wrong).")
        return
    M = match(S, AS, on)
    real = pipeline(S, M, dev, chk)
    G = real["g_dev"].max() + 1

    # chance: shuffled weight rows, same firings
    w9 = W["w"].reshape(256, 9) * W["sc"]
    centre = P["patch"][:, :, 2:5, 2:5].reshape(n, 256, 9)                  # n101's own 3 x 3 window
    chk_ok = np.abs(np.einsum("ct,nct->nc", w9, centre) - S).max()
    print(f"split recomputed from patch: max abs diff {chk_ok:.2e}")
    sh_cos, sh_cp = np.empty((nsh, len(chk))), np.empty((nsh, len(chk)))
    for s in range(nsh):
        pi = np.random.default_rng(1000 + s).permutation(256)
        Ss = np.einsum("ct,nct->nc", w9[pi], centre)
        r = pipeline(Ss, match(Ss, AS, on), dev, chk)
        sh_cos[s], sh_cp[s] = r["cos_near"], r["cp_match"]
    q_cos, q_cp = np.quantile(sh_cos, .99, 0), np.quantile(sh_cp, .99, 0)

    in_group = real["cos_near"] > q_cos
    n_dev = np.bincount(real["g_dev"], minlength=G)
    n_chk = np.bincount(real["near"][in_group], minlength=G)                # CHECK firings IN the group
    holds = (n_dev >= MIN_G) & (n_chk >= MIN_G)
    cp_pass = real["cp_match"] > q_cp
    in_hold = in_group & holds[real["near"]]
    bin_a = in_hold & cp_pass
    bin_b = in_hold & ~cp_pass
    bin_c = ~in_hold
    m = len(chk)
    print(f"\nCHECK firings: {m}.  Groups on DEV: {G} (DEV sizes {n_dev.tolist()}; CHECK firings in each {n_chk.tolist()})")
    print(f"  (a) explained  [{bar(bin_a.mean())}] {bin_a.sum():3d}  {bin_a.mean():.2f}")
    print(f"  (b) named gap  [{bar(bin_b.mean())}] {bin_b.sum():3d}  {bin_b.mean():.2f}")
    print(f"  (c) unexplained[{bar(bin_c.mean())}] {bin_c.sum():3d}  {bin_c.mean():.2f}")
    print(f"  subcircuits cover n101's read (bin c <= .05): {'YES' if bin_c.mean() <= .05 else 'no'}"
          f"{'  | NOT covered by this method (bin c > .25)' if bin_c.mean() > .25 else ''}")
    print(f"  read closed IN A's TERMS (bin a >= .95): {'YES' if bin_a.mean() >= .95 else 'no'}")
    for lv in (.5, .7):
        print(f"  bin (a) with counterpart match also >= {lv}: {(bin_a & (real['cp_match'] >= lv)).mean():.2f}")
    Mc, onc = M[chk], on[chk]
    cpj = real["cp"][real["near"]]
    mine = Mc[np.arange(m), cpj]
    rank = np.array([(Mc[i][onc[i]] > mine[i]).sum() + 1 if onc[i, cpj[i]] else -1 for i in range(m)])
    okr = rank > 0
    print(f"  counterpart ON at the firing: {okr.mean():.2f}; its rank among the A neurons ON there, median "
          f"{int(np.median(rank[okr])) if okr.any() else -1} of median {int(np.median(onc.sum(1)))} ON")
    print(f"  P1: median counterpart match on CHECK {np.median(real['cp_match']):.2f} (bar .7); "
          f"median shuffled 99th pct {np.median(q_cp):.2f}")
    print(f"  in a group (nearest-centroid cosine above own 99th pct): {in_group.mean():.2f}; "
          f"real median {np.median(real['cos_near']):.2f}, shuffled 99th median {np.median(q_cos):.2f}")

    src = P["src"]
    for k in range(G):
        dm = dev[real["g_dev"] == k]
        cm = chk[real["near"] == k]
        Sk = S[dm].mean(0)
        drv = np.argsort(-Sk)[:6]
        sup = np.argsort(Sk)[:4]
        sk = src[dm].mean(0)
        top_src = np.argsort(-np.abs(sk))[:8]
        top3 = np.argsort(-real["cps"][k])[:3]
        kids = [int(j) for j in np.argsort(-real["cps"][k])[:10] if int(j) in CHILDREN]
        tag = "holds" if holds[k] else "does not hold"
        ck = real["near"] == k
        print(f"\nsubcircuit {k}: DEV {len(dm)}, CHECK {len(cm)} ({tag}); CHECK bins a/b/c "
              f"{bin_a[ck].sum()}/{bin_b[ck].sum()}/{bin_c[ck].sum()}; mean n101 input {P['pre'][dm].mean():.2f}")
        print("  drivers     " + ", ".join(f"s3:{c} {Sk[c]:+.3f}" for c in drv))
        print("  suppressors " + ", ".join(f"s3:{c} {Sk[c]:+.3f}" for c in sup))
        print("  sources     " + ", ".join(f"{SRC_NAMES[j]} {sk[j]:+.3f}" for j in top_src))
        print("  counterparts " + ", ".join(f"A n{j} {real['cps'][k][j]:.2f}" for j in top3)
              + f"; CHECK match median {np.median(real['cp_match'][ck]) if ck.any() else float('nan'):.2f}")
        print(f"  children of s3:69 among its top-10 counterparts: {kids if kids else 'none'}")
    np.savez(os.path.join(R, f"oc_subcircuits_{pole}{'_smoke' if smoke else ''}.npz"), dev=dev, chk=chk,
             g_dev=real["g_dev"], near=real["near"], cos_near=real["cos_near"], cp=real["cp"], cps=real["cps"],
             cp_match=real["cp_match"], q_cos=q_cos, q_cp=q_cp, bin_a=bin_a, bin_b=bin_b, bin_c=bin_c, holds=holds,
             rank=rank, in_group=in_group,
             on=on, M=M.astype(np.float16))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(a[0] if a else "pos", int(a[1]) if len(a) > 1 else NSH, "--smoke" in sys.argv)
