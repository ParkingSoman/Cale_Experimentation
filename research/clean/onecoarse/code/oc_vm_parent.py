# scale: neurons=s3:69, A:children, A:block0-all; net=A; photos=saved; where=mac
# rung: Q=vector-math track step 2: is the parent the one shared read direction of its children (Shonu's definition); N=11 R1 children + 200 random 11-sets of block-0 neurons, weights only; EST=1 min Mac; PREV=results/oc_reads_w.npz; PERMIT=Shonu approved plan "New track: deletion as vector math" 2026-11-24 incl. its SCALE-UP REQUEST
"""VECMATH step 2 (2026-11-24). Weights only, numpy, Mac.

Shonu's definition: the parent's read is the component that, removed from the children's reads, leaves those reads
orthogonal to each other. In vectors (unit reads r_i, Gram G = r_i . r_j):
    q_i = r_i - c_i p,  |p| = 1,  c_i = r_i . p   ->   q_i . q_j = G_ij - c_i c_j
So the children's reads have one parent exactly when G off the diagonal = c c^T (one shared direction).
Fit: c = argmin sum_{i != j} (G_ij - c_i c_j)^2 (rank-1 fit of the off-diagonal; diagonal refilled until stable).
Parent direction inside the children's span: p = R^T (R R^T)^-1 c (minimum norm with R p = c), reported normalised.

Predictions (written in the plan before this run):
  P1 p's largest channel is s3:69, with at least half of p's energy
  P2 after removing p, sibling read overlaps fall to the random-pair level (.022)
  P3 the shared part of the 11 siblings is above the 99th pct of 200 random 11-sets of block-0 neurons
  P4 the same fit on the 11 write vectors finds no shared write direction above random
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
P = 69


def bn_scale(bn):
    g, _, _, v = bn
    return g / np.sqrt(v + 1e-5)


def unit(M):
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def off(C):
    return C[~np.eye(len(C), dtype=bool)]


def rank1_off(G, iters=500):
    """c minimising sum_{i!=j} (G_ij - c_i c_j)^2: top eigenvector with the diagonal refilled by c_i^2."""
    H = G.copy()
    np.fill_diagonal(H, 0.0)
    c = np.zeros(len(G))
    for _ in range(iters):
        np.fill_diagonal(H, c ** 2)
        w, U = np.linalg.eigh(H)
        c_new = U[:, -1] * np.sqrt(max(w[-1], 0.0))
        if np.abs(np.abs(c_new) - np.abs(c)).max() < 1e-10:
            c = c_new
            break
        c = c_new
    if c.sum() < 0:
        c = -c
    return c


def fit(M):
    """M: n x dim raw vectors. Returns stats of the one-shared-direction fit on unit vectors."""
    U = unit(M)
    G = U @ U.T
    c = rank1_off(G)
    o = off(G)
    resid = off(G - np.outer(c, c))
    p = U.T @ np.linalg.solve(G + 1e-9 * np.eye(len(G)), c)   # minimum-norm p with U p = c
    pnorm = np.linalg.norm(p)
    # Exact form of the definition: a UNIT p with r_i . p = c_i. It exists when |p_min| <= 1 (add a part orthogonal
    # to every r_i). Then q_i . q_j = G_ij - c_i c_j exactly, and |q_i|^2 = 1 - c_i^2.
    Qd = G - np.outer(c, c)
    nq = np.sqrt(np.clip(np.diag(Qd), 1e-12, None))
    Qc = Qd / np.outer(nq, nq)
    pc = (p.reshape(-1, 9) ** 2).sum(1) if p.size % 9 == 0 else None
    top_share = float(pc.max() / pc.sum()) if pc is not None else np.nan
    return dict(top_share=top_share, c=c, mean_off=o.mean(), median_off=np.median(o), shared=off(np.outer(c, c)).mean(),
                explained=1 - (resid ** 2).sum() / (o ** 2).sum(), p=p / max(pnorm, 1e-12), pnorm=pnorm,
                after_median=np.median(off(Qc)), after_mean=off(Qc).mean())


def main():
    w = np.load(os.path.join(HERE, "..", "..", "neuron", "results", "coarse", "r18", "w_A.npz"))
    dial = np.load(os.path.join(RES, "oc_dial.npz"))
    s1, s2 = bn_scale(w["bn1_0"]), bn_scale(w["bn2_0"])
    R = w["W1_0"] * s1[:, None, None, None]                     # 512 x 256 x 3 x 3 (read kernels)
    V = w["W2_0"] * s2[:, None, None, None]                     # out 512 x in 512 x 3 x 3; V[:, i] = write of i
    wnorm = np.sqrt((V ** 2).sum((0, 2, 3)))
    order = np.argsort(-(R[:, P].sum((1, 2)) * wnorm))
    assert (order[:30] == dial["order"][:30]).all(), "R1 order differs from oc_dial.npz"
    kids = order[:11]
    Rall = R.reshape(512, -1)
    Vall = V.transpose(1, 0, 2, 3).reshape(512, -1)

    rng = np.random.default_rng(0)
    pool = np.setdiff1d(np.arange(512), order[:30])
    sets = [rng.choice(pool, 11, replace=False) for _ in range(200)]

    out = {}
    for name, M in (("read", Rall), ("write", Vall)):
        k = fit(M[kids])
        rnd = [fit(M[s]) for s in sets]
        out[name] = (k, rnd)
        print(f"\n== {name}s ==")
        rows = ["mean_off", "median_off", "shared", "explained", "after_median", "after_mean", "pnorm", "top_share"]
        print(f"{'stat':14s} {'siblings':>9s} {'random med':>11s} {'random 99th':>12s} {'sib pct':>8s}")
        for r in rows:
            rv = np.array([x[r] for x in rnd])
            pct = (rv < k[r]).mean() * 100
            print(f"{r:14s} {k[r]:9.4f} {np.median(rv):11.4f} {np.percentile(rv, 99):12.4f} {pct:7.1f}%")
        print("c (each child's overlap with p):", " ".join(f"n{n}:{v:.3f}" for n, v in zip(kids, k["c"])))

    # P1: what channel is the read parent p? (p lives in 256 x 3 x 3)
    p = out["read"][0]["p"].reshape(256, 3, 3)
    e = (p ** 2).sum((1, 2))
    e = e / e.sum()
    top = np.argsort(-e)[:8]
    print("\nP1: read parent p, channel energy share:", " ".join(f"s3:{t} {e[t]:.3f}" for t in top))
    print("    s3:69 rank", int((e > e[P]).sum()), "share", round(float(e[P]), 3),
          "| p at s3:69, 3x3 (sign):", np.round(p[P] / np.abs(p[P]).max(), 2).tolist())
    # also against the child mean read and the bare channel-69 direction
    e69 = np.zeros_like(p); e69[P] = np.sign(R[kids, P].mean(0)) * np.abs(R[kids, P]).mean(0)
    cosp = float((p.ravel() @ e69.ravel()) / np.linalg.norm(e69))
    print("    cos(p, children's mean channel-69 kernel placed alone):", round(cosp, 3))

    # P4: the write parent, if any: which output channels does it use?
    pw = out["write"][0]["p"].reshape(512, 3, 3)
    ew = (pw ** 2).sum((1, 2)); ew = ew / ew.sum()
    print("P4: write p, top output channels:", " ".join(f"s4:{t} {ew[t]:.3f}" for t in np.argsort(-ew)[:6]))

    # Is the children's shared write the parent's own write? The parent writes into the stage-4 stream through its
    # 1x1 skip column Wd[:, 69] (bn folded). Compare with the shared write summed over the 3 x 3 offsets.
    sd = bn_scale(w["bnd"])
    Wd = w["Wd"][:, :, 0, 0] * sd[:, None]
    pw_sum = pw.sum((1, 2))
    skip = Wd[:, P]
    cs = float(pw_sum @ skip / np.linalg.norm(pw_sum) / np.linalg.norm(skip))
    rnd_cs = np.array([float(pw_sum @ Wd[:, j] / np.linalg.norm(pw_sum) / np.linalg.norm(Wd[:, j])) for j in range(256)])
    print("    cos(shared write summed over 3x3, parent's skip write Wd[:,69]):", round(cs, 3),
          "| rank among 256 stage-3 skip columns:", int((rnd_cs > cs).sum()), "| best:",
          " ".join(f"s3:{j} {rnd_cs[j]:.3f}" for j in np.argsort(-rnd_cs)[:5]))
    # and the children's MEAN write (the trivial shared direction)
    mw = unit(Vall[kids]).mean(0); mw /= np.linalg.norm(mw)
    print("    cos(shared write p, plain mean of the 11 unit writes):", round(float(out['write'][0]['p'] @ mw), 3))

    np.savez(os.path.join(RES, "oc_vm_parent.npz"), kids=kids, sets=np.array(sets),
             read_c=out["read"][0]["c"], read_p=out["read"][0]["p"], write_c=out["write"][0]["c"],
             write_p=out["write"][0]["p"],
             read_rand=np.array([[x[r] for r in ("mean_off", "shared", "explained", "after_median")] for x in out["read"][1]]),
             write_rand=np.array([[x[r] for r in ("mean_off", "shared", "explained", "after_median")] for x in out["write"][1]]),
             ch_energy=e)


if __name__ == "__main__":
    main()
