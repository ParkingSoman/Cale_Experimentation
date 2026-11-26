# scale: neurons=A:block0-all; net=A, C-ladder, L-ladder, WC-ladder, R4C-ladder; photos=saved, broden-held; where=mac
# rung: Q=plan "Read what training adds on top of the calculation", step 3 controls (advisor review): shuffled-unit rank, chance count, plus the leftover plan items (G top-5 read, n430 K5b check); N=weights + vecread dictionary; EST=2 min Mac; PERMIT=Shonu approved plan 2026-11-25
"""Controls for the widening test (oc_vs_read.py), added after the first reading.
1. Unit rank: for the named partner p, rank the named unit j among ALL kept units by (a) cos(write change, p's A write)
   and (b) cos(read-pattern change, p's read pattern). A common direction (p moves every unit) gives j a low rank.
2. Chance count: kept units with SOME deleted partner in the top 5% of both its write list and its read list, and
   a >= .8. Shuffled baseline: the read list's partner labels permuted (20 shuffles).
3. Leftover plan items: vecread reads of the readers' Fisher G's top 5 eigenvectors (s4b0, write metric, both signs
   equal so the top concept by |cos| is reported); check that n430's own A write reads as in K5b tools.
Output: results/oc_vs/widen_ctl.md"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "vecread"))
import vecread as vr  # noqa: E402
from oc_vs_read import R18, LAD, OUT, eff  # noqa: E402

RNG = np.random.default_rng(11)


def unit_rows(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-30)


def main():
    dk, dic = vr.Dictionary("s3k"), vr.Dictionary("s4b0")
    pat = lambda R: dk.metric(np.atleast_2d(R) @ dk.cov.T, True)          # whitened patterns, rows
    A = np.load(os.path.join(R18, "w_A.npz"))
    VA, SA, RA, _ = eff(A)
    PA = unit_rows(pat(RA))
    VAu = unit_rows(VA)
    md = ["# Widening controls and leftover plan items (oc_vs_widen_ctl.py)", "",
          "Note: the read bar in oc_vs_read.py first used raw read kernels; it was changed to whitened patterns (the "
          "vecread rule) after that first run. With raw kernels, the 384 rows read .22 / .46 / .76 (fail); n339 at 64 "
          "passed with both metrics (.98).", "",
          "## 1. Rank of the named unit among all kept units, for the same partner", "",
          "| Cut | Unit -> partner | Build | Write rank among kept | Read rank among kept |", "|---|---|---|---|---|"]
    named = {64: [(339, 476)], 192: [(430, 114)], 384: [(430, 501), (76, 374), (213, 501)], 11: [(5, 464)]}
    count_rows = []
    for n in (11, 64, 192, 384):
        d = os.path.join(LAD, f"n{n}")
        jt = np.load(os.path.join(d, "jobtest_L.npz"))
        D, K = jt["D"], jt["K"]
        for b in ("L", "R4C"):
            Vn, _, Rn, _ = eff(np.load(os.path.join(d, f"w_{b}.npz")))
            dW = unit_rows(Vn[K] - VA[K])
            dRraw = Rn[K] - RA[K]
            moved = np.linalg.norm(dRraw, axis=1) > 1e-9
            dP = unit_rows(pat(dRraw))
            CW = dW @ VAu[D].T                     # kept x deleted
            CR = dP @ PA[D].T
            for j, p in named[n]:
                ij, ip = int(np.nonzero(K == j)[0][0]), int(np.nonzero(D == p)[0][0])
                rw = float((CW[:, ip] < CW[ij, ip]).mean())
                rr = float((CR[moved, ip] < CR[ij, ip]).mean())
                md.append(f"| {n} | n{j} -> n{p} | {b} | {rw:.2f} | {rr:.2f} |")
            # chance count
            a = np.array([np.linalg.lstsq(VA[k][:, None], Vn[k], rcond=None)[0][0] for k in K])
            q = max(1, int(np.ceil(.05 * len(D))))
            topW = np.argsort(-CW, 1)[:, :q]
            topR = np.argsort(-CR, 1)[:, :q]

            def count(tR):
                hit = np.array([len(set(topW[i]) & set(tR[i])) > 0 for i in range(len(K))])
                return int((hit & (a >= .8) & moved).sum())

            real = count(topR)
            sh = [count(np.array([RNG.permutation(len(D))[t] for t in topR])) for _ in range(20)]
            count_rows.append(f"| {n} | {b} | {real} of {int(moved.sum())} | {np.mean(sh):.1f} (max {max(sh)}) |")
    md += ["", "## 2. Kept units that pass write + read + keeps-job with some partner (top 5% of the deleted on both)", "",
           "| Cut | Build | Real | Shuffled partner labels, mean (max of 20) |", "|---|---|---|---|"] + count_rows

    # leftover items
    G = np.load(os.path.join(OUT, "G.npy"))
    ev, U = np.linalg.eigh(G)
    md += ["", "## 3. Readers' Fisher G (stream after block 0), top 5 eigenvectors read as writes", "",
           f"Top-20 share of trace {ev[-20:].sum() / ev.sum():.3f}; top-5 share {ev[-5:].sum() / ev.sum():.3f}.", "",
           "| Rank | Share | + pole (up 3) | - pole (down 3) | Leftover |", "|---|---|---|---|---|"]
    for i in range(1, 6):
        c = vr.read_vector(dic, U[:, -i], "write")
        md.append(f"| {i} | {ev[-i] / ev.sum():.3f} | " + ", ".join(f"{x} {s:+.2f}" for x, _, s in c["up"][:3]) +
                  " | " + ", ".join(f"{x} {s:+.2f}" for x, _, s in c["down"][:3]) + f" | {c['leftover']:.2f} |")
    c = vr.read_vector(dic, SA[430], "write")
    md += ["", f"Check: n430's A write reads {', '.join(x for x, _, _ in c['up'][:3])} "
               f"(tool used for all 512 A writes in K5b; top cosine {c['up'][0][2]:.2f}, leftover {c['leftover']:.2f})."]
    open(os.path.join(OUT, "widen_ctl.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
