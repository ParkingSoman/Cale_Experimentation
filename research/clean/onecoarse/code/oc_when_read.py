# scale: neurons=s3:69, A:children; net=A; photos=saved; where=mac
# rung: Q=read Part B of "Reads: why siblings split" (W0-W3); N=11 children, 3 pairs; EST=seconds; PREV=results/oc_when.npz, results/oc_dial.npz; PERMIT=Shonu approved plan 2026-11-24
"""Scores W0-W3 of LOG "Reads: why siblings split" from results/oc_when.npz and results/oc_dial.npz."""
import os
import sys

import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def rank(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r


def spearman(a, b):
    return np.corrcoef(rank(np.asarray(a)), rank(np.asarray(b)))[0, 1]


def main(path):
    w = np.load(path)
    d = np.load(os.path.join(RES, "oc_dial.npz"))
    kids, pairs = w["kids"].tolist(), [tuple(p) for p in w["pairs"]]
    pool = d["pool"].tolist()
    S1, S2, N = w["S1"], w["S2"], float(w["N"])
    mu = S1 / N
    cov = S2 / N - np.outer(mu, mu)
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    mis_c = 1 - corr ** 2
    mis_u = 1 - S2 ** 2 / np.outer(np.diag(S2), np.diag(S2))
    ix = {n: a for a, n in enumerate(kids)}
    rows = len(w["dev_rows"])

    # W0
    e = []
    for i, _ in pairs:
        p = pool.index(i)
        e.append(np.abs(w[f"dev_del_i_{i}_None"] - d["one_delete"][p][:rows]).max())
        e.append(np.abs(w[f"dev_morph_i_{i}_None"] - d["one_morph"][p][:rows]).max())
    print(f"W0: max diff from oc_dial.npz {max(e):.2e} (bar 2e-3) -> {'pass' if max(e) < 2e-3 else 'FAIL'}")

    # W1: sibling WHEN mismatch at the cell
    iu = np.triu_indices(11, 1)
    mc, mu_ = mis_c[:11, :11][iu], mis_u[:11, :11][iu]
    o = np.argsort(mc)[:5]
    print(f"\nW1: sibling mismatch (1 - corr^2), 55 pairs: min {mc.min():.3f}, median {np.median(mc):.3f}; "
          f"uncentred min {mu_.min():.3f}, median {np.median(mu_):.3f}")
    print("  lowest 5: " + ", ".join(f"n{kids[iu[0][k]]}-n{kids[iu[1][k]]} {mc[k]:.3f}" for k in o))
    print(f"  -> {'pass' if mc.min() >= .7 else 'FAIL'} (bar: all >= .7)")
    print("  each child against the parent's push through it (1 - corr^2 / uncentred): " + ", ".join(
        f"n{n} {mis_c[a, 11 + a]:.2f}/{mis_u[a, 11 + a]:.2f}" for a, n in enumerate(kids)))

    # W2 and the build table
    print("\n| Pair (i -> j) | c_a | c_b | care_i (DEV KL del_i) | build | mismatch (c / u) | DEV KL | rec. te_rand | rec. te_strong |")
    print("|---|---|---|---|---|---|---|---|---|")
    pred_c, pred_u, meas = [], [], []
    for p, (i, j) in enumerate(pairs):
        ai, aj = ix[i], ix[j]
        care = w[f"dev_del_i_{i}_None"].mean()
        carriers = {"morph_i": (11 + ai, f"morph_i_{i}_None"), "hand_a": (aj, f"hand_a_{i}_{j}"),
                    "hand_b": (22 + p, f"hand_b_{i}_{j}")}
        for b, (v, key) in carriers.items():
            kl = w[f"dev_{key}"].mean()
            rr = 1 - w[f"te_rand_{key}"].mean() / w[f"te_rand_del_i_{i}_None"].mean()
            rs = 1 - w[f"te_strong_{key}"].mean() / w[f"te_strong_del_i_{i}_None"].mean()
            pred_c.append(care * mis_c[ai, v]); pred_u.append(care * mis_u[ai, v]); meas.append(kl)
            print(f"| n{i} -> n{j} | {w['c'][p, 0]:.3f} | {w['c'][p, 1]:.3f} | {care:.5f} | {b} | "
                  f"{mis_c[ai, v]:.3f} / {mis_u[ai, v]:.3f} | {kl:.5f} | {rr:.3f} | {rs:.3f} |")
    s_c, s_u = spearman(pred_c, meas), spearman(pred_u, meas)
    print(f"\nW2: Spearman(care x mismatch, DEV KL) over 9: centred {s_c:.2f}, uncentred {s_u:.2f} "
          f"-> {'pass' if s_c >= .6 else 'FAIL'} (bar .6, centred as written)")

    # W3
    rng = np.random.default_rng(0)
    print("\nW3:")
    for p, (i, j) in enumerate(pairs):
        a = w[f"te_rand_hand_a_{i}_{j}"]; b = w[f"te_rand_hand_b_{i}_{j}"]; dl = w[f"te_rand_del_i_{i}_None"]
        diffs = []
        for _ in range(2000):
            s = rng.integers(0, len(dl), len(dl))
            diffs.append((1 - b[s].mean() / dl[s].mean()) - (1 - a[s].mean() / dl[s].mean()))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        top = w["dev_j_peak"][:, p] >= np.percentile(w["dev_j_peak"][:, p], 95)
        extra = w[f"dev_hand_b_{i}_{j}"][top].mean() - w[f"dev_del_i_{i}_None"][top].mean()
        worth = w[f"dev_del_j_None_{j}"][top].mean()
        ok = lo > 0 and extra <= .1 * worth
        print(f"  n{i} -> n{j}: rec(hand_b) - rec(hand_a), te_rand: [{lo:.3f}, {hi:.3f}]; on n{j}'s top 5% "
              f"({top.sum()} photos): KL(hand_b) - KL(del_i) = {extra:.5f}, .1 x KL(del_j) = {.1 * worth:.5f} "
              f"-> {'pass' if ok else 'FAIL'}")
    print("\nwoken carrier fires (photo peak > 0) on share of DEV photos: " + ", ".join(
        f"n{j}' {np.mean(w['dev_woken_peak'][:, p] > 0):.2f} (n{j} {np.mean(w['dev_j_peak'][:, p] > 0):.2f})"
        for p, (_, j) in enumerate(pairs)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, "oc_when.npz"))
