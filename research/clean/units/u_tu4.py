# serves: TU4 (UNIT-BRANCH.md)
"""Theorem TU4 on random linear one-hidden-layer instances with a fixed metric F.
Least single box at budget k with a free constant = reduced-rank regression in the F metric:
   R* = C_S - F^(-1/2) [rank-k part of F^(1/2) C_S Sigma_SS^(1/2)] Sigma_SS^(-1/2).
Part 1: Sigma_ST = 0 (built exactly) -> cross term tr(R_S^T F R_T Sigma_TS) = 0 for random R; joint by alternating
        exact block minimisation from many starts never goes below the sum of single optima.
Part 2: Sigma_ST != 0, k_T >= rank(C_T) + n_y: joint value with R_T = -G* equals single_S - |Pi|^2 (closed form), < single_S.
"""
import json
import os

import numpy as np

rng = np.random.default_rng(2)


def msqrt(A, inv=False):
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 1e-15, None)
    return (V * (w ** (-.5 if inv else .5))) @ V.T


def rrr(CS, F, Sig, k):
    """least E|F^(1/2) R z|^2 over R = CS - rank-k; returns R*, value."""
    Fh, Fhi, Sh, Shi = msqrt(F), msqrt(F, True), msqrt(Sig), msqrt(Sig, True)
    U, s, Vt = np.linalg.svd(Fh @ CS @ Sh)
    lowk = (U[:, :k] * s[:k]) @ Vt[:k]
    R = CS - Fhi @ lowk @ Shi
    return R, float(np.trace(R.T @ F @ R @ Sig))


out = {"part1": [], "part2": []}
for trial in range(30):
    ny, nS, nT, npts = 3, 4, 3, 200
    F = (lambda Q: Q @ Q.T + .1 * np.eye(ny))(rng.normal(size=(ny, ny)))
    CS, CT = rng.normal(size=(ny, nS)), rng.normal(size=(ny, nT))
    # part 1: z_S and z_T with exactly zero sample cross-covariance
    zS = rng.normal(size=(npts, nS)); zS -= zS.mean(0)
    zT = rng.normal(size=(npts, nT)); zT -= zT.mean(0)
    zT -= zS @ np.linalg.lstsq(zS, zT, rcond=None)[0]
    SS, TT, ST = zS.T @ zS / npts, zT.T @ zT / npts, zS.T @ zT / npts
    kS, kT = 1, 1
    RS, vS = rrr(CS, F, SS, kS); RT, vT = rrr(CT, F, TT, kT)
    cross = max(abs(np.trace(A.T @ F @ B @ ST.T)) for A, B in ((rng.normal(size=(ny, nS)), rng.normal(size=(ny, nT))) for _ in range(20)))
    # joint: minimise E|F^(1/2)(R_S zS + R_T zT)|^2 over rank-k corrections by alternating exact steps
    best = np.inf
    for start in range(10):
        A = CS - rng.normal(size=(ny, kS)) @ rng.normal(size=(kS, nS))
        B = CT - rng.normal(size=(ny, kT)) @ rng.normal(size=(kT, nT))
        for it in range(50):
            # minimise over A given B: target for S is CS + (B zT predicted by zS) -> zero here since ST = 0
            tgt = CS + B @ ST.T @ np.linalg.inv(SS)
            A0, _ = rrr(tgt, F, SS, kS); A = A0 - (tgt - CS)
            tgt = CT + A @ ST @ np.linalg.inv(TT)
            B0, _ = rrr(tgt, F, TT, kT); B = B0 - (tgt - CT)
        val = np.mean(np.einsum("ni,ij,nj->n", zS @ A.T + zT @ B.T, F, zS @ A.T + zT @ B.T))
        best = min(best, val)
    out["part1"].append(dict(max_cross=float(cross), joint=float(best), sum_single=vS + vT, I=float(best - vS - vT)))
    # part 2: correlated activity
    zT2 = zT + zS @ rng.normal(size=(nS, nT)); zT2 -= zT2.mean(0)
    TT2, ST2 = zT2.T @ zT2 / npts, zS.T @ zT2 / npts
    RS, vS = rrr(CS, F, SS, kS)
    # best G: minimise E|F^(1/2)(RS zS - G zT2)|^2 -> G = RS ST2 TT2^-1
    G = RS @ ST2 @ np.linalg.inv(TT2)
    E = zS @ RS.T - zT2 @ G.T
    joint = float(np.mean(np.einsum("ni,ij,nj->n", E, F, E)))
    Pi2 = float(np.trace(G.T @ F @ G @ TT2))
    out["part2"].append(dict(single_S=vS, joint_upper=joint, Pi2=Pi2, check=float(joint - (vS - Pi2)),
                             FRSigma_norm=float(np.linalg.norm(F @ RS @ ST2))))
p1, p2 = out["part1"], out["part2"]
out["summary"] = dict(p1_max_cross=max(r["max_cross"] for r in p1), p1_max_absI=max(abs(r["I"]) for r in p1),
                      p2_max_formula_err=max(abs(r["check"]) for r in p2), p2_all_negative=all(r["Pi2"] > 1e-9 for r in p2))
print(json.dumps(out["summary"], indent=1))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "u_tu4.json"), "w"), indent=1)
