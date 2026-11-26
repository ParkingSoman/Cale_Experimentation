# scale: neurons=s3:69, A:children, A:block1-weights; net=A; photos=saved; where=mac
# rung: Q=reads from weights: do siblings and their readers read orthogonal directions, and what does a parent rewrite leak; N=11 R1 children + block-1 read weights; EST=1 min Mac; PREV=plan "Reads: why siblings split" 2026-11-24; PERMIT=Shonu approved plan
"""Part A of LOG "Reads: why siblings split" (2026-11-24). Weights only, numpy, Mac.

Objects (bn folded everywhere, rows of w_A bn = weight, bias, running mean, running var):
  R_i  child i's read kernel on the stage-3 stream, 256 x 3 x 3 (stride 2)
  V_i  child i's write column into the stage-4 stream, 512 x 3 x 3
  K_qi block-1 neuron q's read of child i's write: the composed 5 x 5 map sum_c W1_1[q,c] (x) V[c,i]
       (two cross-correlations compose into one with kernel K[a+b] = W[a] V[b]; the block-0 output ReLU is ignored)
A1 reader map, A2 private read (outside the span of the 10 siblings' reads), A3 the leak of a parent write that
wakes one kept child (best direction = least read by the other 511 block-0 neurons).
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
P = 69


def bn_scale(bn):
    g, _, _, v = bn
    return g / np.sqrt(v + 1e-5)


def cos_mat(M):
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    return M @ M.T


def off(C):
    return C[~np.eye(len(C), dtype=bool)]


def compose(W, V):
    """W: readers x 512 x 3 x 3, V: 512 x n x 3 x 3 -> readers x n x 5 x 5."""
    K = np.zeros((W.shape[0], V.shape[1], 5, 5))
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    K[:, :, a + c, b + d] += W[:, :, a, b] @ V[:, :, c, d]
    return K


def main():
    w = np.load(os.path.join(HERE, "..", "..", "neuron", "results", "coarse", "r18", "w_A.npz"))
    dial = np.load(os.path.join(RES, "oc_dial.npz"))
    s1, s2, s1b = bn_scale(w["bn1_0"]), bn_scale(w["bn2_0"]), bn_scale(w["bn1_1"])
    R = w["W1_0"] * s1[:, None, None, None]                     # 512 x 256 x 3 x 3
    V = w["W2_0"] * s2[:, None, None, None]                     # out 512 x in 512 x 3 x 3; V[:, i] = write of i
    W1 = w["W1_1"] * s1b[:, None, None, None]                   # block-1 readers
    sd = bn_scale(w["bnd"])
    Wd = w["Wd"][:, :, 0, 0] * sd[:, None]                      # 1x1 skip, 512 x 256

    # A0: the R1 order must match oc_dial.npz
    wnorm = np.sqrt((V ** 2).sum((0, 2, 3)))
    score = R[:, P].sum((1, 2)) * wnorm
    order = np.argsort(-score)
    assert (order[:30] == dial["order"][:30]).all(), "R1 order differs from oc_dial.npz"
    kids = order[:11]
    rng = np.random.default_rng(0)
    rnd = rng.choice(np.setdiff1d(np.arange(512), order[:30]), 200, replace=False)
    print("A0: R1 order matches oc_dial.npz. Children:", kids.tolist())

    # first-look table, full kernels
    Wk, Wr = V[:, kids].transpose(1, 0, 2, 3).reshape(11, -1), V[:, rnd].transpose(1, 0, 2, 3).reshape(200, -1)
    Rk, Rr = R[kids].reshape(11, -1), R[rnd].reshape(200, -1)
    Rk0 = R[kids].copy(); Rk0[:, P] = 0; Rk0 = Rk0.reshape(11, -1)
    Kk, Kr = compose(W1, V[:, kids]), compose(W1, V[:, rnd])    # 512 x n x 5 x 5
    Gk, Gr = Kk.transpose(1, 0, 2, 3).reshape(11, -1), Kr.transpose(1, 0, 2, 3).reshape(200, -1)
    strength_k, strength_r = (Kk ** 2).sum((2, 3)), (Kr ** 2).sum((2, 3))   # 512 x n
    part = lambda s: s.sum(0) ** 2 / (s ** 2).sum(0)
    rows = [
        ("write cos, median abs", np.median(np.abs(off(cos_mat(Wk)))), np.abs(off(cos_mat(Wk))).max(),
         np.median(np.abs(off(cos_mat(Wr))))),
        ("read cos (stage 3), median", np.median(off(cos_mat(Rk))), off(cos_mat(Rk)).max(), np.median(off(cos_mat(Rr)))),
        ("read cos without ch 69, median", np.median(off(cos_mat(Rk0))), off(cos_mat(Rk0)).max(), np.nan),
        ("reader-profile cos (5x5 composed), median", np.median(off(cos_mat(Gk))), off(cos_mat(Gk)).max(),
         np.median(off(cos_mat(Gr)))),
    ]
    print("\n| Item | Siblings median | Siblings max | Random median |\n|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f} |")
    print(f"| readers per child (participation of 512) | {np.median(part(strength_k)):.0f} "
          f"({part(strength_k).min():.0f}-{part(strength_k).max():.0f}) | | {np.median(part(strength_r)):.0f} |")
    ch69 = (R[kids, P] ** 2).sum((1, 2)) / (R[kids] ** 2).sum((1, 2, 3))
    print("ch 69 share of each child's read energy:", np.round(ch69, 3).tolist())

    # A1: named readers per child, and shared top readers per pair
    signed = Kk.sum((2, 3))                                     # 512 x 11: response to a flat unit of child i
    top = {}
    print("\nA1: top 5 block-1 readers per child (strength share; sign of the flat response)")
    for j, n in enumerate(kids):
        s = strength_k[:, j]
        o = np.argsort(-s)[:5]
        top[n] = set(np.argsort(-s)[:20].tolist())
        print(f"  n{n}: " + ", ".join(f"b1:{q} {s[q] / s.sum():.3f}{'+' if signed[q, j] > 0 else '-'}" for q in o)
              + f"   top-20 hold {np.sort(s)[::-1][:20].sum() / s.sum():.2f}")
    shared = np.array([[len(top[a] & top[b]) for b in kids] for a in kids])
    rtop = [set(np.argsort(-strength_r[:, j])[:20].tolist()) for j in range(200)]
    rshared = [len(rtop[a] & rtop[b]) for a in range(0, 200, 2) for b in range(1, 200, 2)][:2000]
    print(f"shared top-20 readers per sibling pair: median {np.median(off(shared)):.0f}, max {off(shared).max()}; "
          f"random pairs median {np.median(rshared):.0f}, 99th pct {np.percentile(rshared, 99):.0f}")
    pi, pj = np.unravel_index(np.argmax(np.where(np.eye(11, dtype=bool), -1, shared)), shared.shape)
    print(f"  most shared pair: n{kids[pi]} - n{kids[pj]} ({shared[pi, pj]} of 20)")

    # A2: private read = part of R_i outside the span of the 10 siblings' reads
    def private(Rset, i):
        others = np.delete(Rset, i, 0)
        Q, _ = np.linalg.qr(others.T)
        r = Rset[i]
        pr = r - Q @ (Q.T @ r)
        return pr, (pr ** 2).sum() / (r ** 2).sum()
    print("\nA2: private read of each child (outside the span of its 10 siblings' reads)")
    priv = {}
    rnd_share = []
    for t in range(20):
        sel = rng.choice(np.setdiff1d(np.arange(512), order[:30]), 11, replace=False)
        rnd_share.append(private(R[sel].reshape(11, -1), 0)[1])
    for j, n in enumerate(kids):
        pr, sh = private(Rk, j)
        priv[n] = pr
        ce = (pr.reshape(256, 9) ** 2).sum(1)
        o = np.argsort(-ce)[:5]
        sign = pr.reshape(256, 9).sum(1)
        print(f"  n{n}: private share {sh:.3f}; ch 69 part of private {ce[P] / ce.sum():.3f}; top channels "
              + ", ".join(f"s3:{c} {ce[c] / ce.sum():.2f}{'+' if sign[c] > 0 else '-'}" for c in o))
    print(f"  random 11-sets, private share of one member: median {np.median(rnd_share):.3f}")
    # pairs whose private reads overlap most (positive cos): candidate pairs for Part B
    Pv = np.stack([priv[n] for n in kids])
    Cp = cos_mat(Pv)
    iu = np.triu_indices(11, 1)
    o = np.argsort(-Cp[iu])[:5]
    print("  private-read overlap, top 5 pairs: " + ", ".join(
        f"n{kids[iu[0][k]]}-n{kids[iu[1][k]]} {Cp[iu][k]:.3f}" for k in o))
    print(f"  private-read overlap, median over 55 pairs {np.median(Cp[iu]):.3f}")

    # A3: a parent write that wakes kept child j. The parent adds p * u to the stage-3 stream (u in R^256, same cell).
    # Neuron k's pre-activation changes by about (r_k . u) p, with r_k = R_k summed over 3x3.
    r = R.sum((2, 3))                                           # 512 x 256
    print("\nA3: parent write u that raises child j's input by 1 per unit of p")
    print("| child j | u = j's read | other 511 neurons: rms change (u = j's read) | best u: rms change | "
          "best u: neurons moved > .25 | skip stream change / j's write norm |\n|---|---|---|---|---|---|")
    a3 = []
    for n in kids:
        rj = r[n]
        others = np.delete(r, n, 0)
        for name, u in (("own", rj), ("best", np.linalg.solve(others.T @ others + 1e-6 * np.eye(256), rj))):
            u = u / (rj @ u)                                    # raise j by exactly 1
            leak = others @ u
            skip = np.linalg.norm(Wd @ u) / np.linalg.norm(V[:, n].sum((1, 2)))
            if name == "own":
                own_rms = np.sqrt((leak ** 2).mean())
            else:
                a3.append((n, own_rms, np.sqrt((leak ** 2).mean()), int((np.abs(leak) > .25).sum()), skip))
    for n, o_, b_, k_, s_ in a3:
        print(f"| n{n} | | {o_:.3f} | {b_:.3f} | {k_} | {s_:.2f} |")
    print("Reference: changing j's OWN read moves no other neuron (0 by construction).")

    np.savez(os.path.join(RES, "oc_reads_w.npz"), kids=kids, shared=shared, private_cos=Cp,
             strength=strength_k, signed=signed, ch69=ch69, a3=np.array([a[1:] for a in a3]))
    print("saved results/oc_reads_w.npz")


if __name__ == "__main__":
    main()
