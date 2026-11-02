# serves: T1
"""Exact synthetic instance of the symmetric model of COARSE-THEORY.md B.1. Checks Theorem T1 to machine precision.

The cyclic group Z_N acts exactly: on inputs by a cyclic shift of the first axis, on members by i -> i+1, on reader
channels by a cyclic shift of their first axis. The input sample is symmetrised over all N shifts, so H-sym holds
exactly for the empirical law. H-clean: member kernels are shifts of one template. H-read: write columns are shifts
of one template column, and the reader metric M is shift-invariant.

Predictions (written before the run, from T1):
  P1  C and Phi are circulant (error ~ 1e-13 relative).
  P2  if the premise of part 3 holds, the best k = 1 read is constant across members (spread ~ 1e-12) and the
      write C1/(1'C1) equals 1/N for every member (error ~ 1e-13).
  P3  the k = 1 Eckart-Young objective equals the best objective found by direct alternating least squares.
  P4  if chat*phihat decreases over n = 0, 1, 2, the best k = 3 read row space equals span{1, cos, sin}
      (largest principal angle ~ 1e-12 rad).
  Controls: breaking H-read (random write columns) or H-sym (unsymmetrised sample) makes P1/P2 fail by O(1).
"""
import numpy as np

rng = np.random.default_rng(0)
N, m, q, S = 10, 6, 4, 4000            # members, input width per angle, reader blocks, samples before symmetrising


def shift(a, k, axis=0):
    return np.roll(a, k, axis=axis)


def build(sym_inputs=True, read_sym=True):
    K0 = np.abs(rng.normal(size=(N, m))) * np.exp(-np.minimum(np.arange(N), N - np.arange(N)) ** 2 / 4.0)[:, None]
    b = -0.5
    X = np.abs(rng.normal(size=(S, N, m))) + 0.3 * rng.normal(size=(S, N, m))
    if sym_inputs:
        X = np.concatenate([shift(X, k, axis=1) for k in range(N)], axis=0)
    Kmem = np.stack([shift(K0, i) for i in range(N)])              # member i kernel = shift^i K0
    Z = np.einsum("snm,inm->si", X, Kmem) + b
    H = np.maximum(Z, 0.0)                                          # (samples, N)
    W0 = rng.normal(size=(N, q))
    if read_sym:
        W = np.stack([shift(W0, i).ravel() for i in range(N)], axis=1)   # (N*q, N): column i = shift^i W0
    else:
        W = rng.normal(size=(N * q, N))
    A = rng.normal(size=(N * q, N * q)); A = A @ A.T
    P = lambda k: np.kron(np.roll(np.eye(N), k, axis=0), np.eye(q))     # reader shift, orthogonal
    M = sum(P(k) @ A @ P(k).T for k in range(N)) / N + np.eye(N * q)  # shift-invariant metric
    return H, W, M


def circ_err(A):
    Ash = np.roll(np.roll(A, 1, axis=0), 1, axis=1)
    return np.abs(Ash - A).max() / np.abs(A).max()


def psd_sqrt(A, inv=False):
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, 1e-300)
    return (V * (w ** (-0.5 if inv else 0.5))) @ V.T


def analyse(label, H, W, M):
    C = H.T @ H / len(H)
    Phi = W.T @ M @ W
    Ms, Ch, Chi = psd_sqrt(M), psd_sqrt(C), psd_sqrt(C, inv=True)
    Y = Ms @ W @ Ch
    U, s, Vt = np.linalg.svd(Y)
    obj_full = 0.5 * np.trace(Phi @ C)
    obj_k1 = obj_full - 0.5 * s[0] ** 2
    v = Vt[0]
    simple = (s[0] ** 2 - s[1] ** 2) / s[0] ** 2
    read = v @ Chi
    read = read / read.mean()
    spread = np.abs(read - 1).max()
    lam_w = C @ np.ones(N) / (np.ones(N) @ C @ np.ones(N))
    write_err = np.abs(lam_w - 1 / N).max() * N
    # P3: alternating least squares for k = 1 from random starts
    best = np.inf
    for _ in range(5):
        lam = rng.normal(size=N)
        for _ in range(300):
            t = H @ lam
            Bv = W @ (C @ lam) / (lam @ C @ lam)                     # best write for this read (T0 part 3)
            # best read for this write: minimise tr((W - B lam^T)^T M (W - B lam^T) C) over lam
            lam = np.linalg.solve(C, (C @ W.T @ M @ Bv)) / (Bv @ M @ Bv)
        E = W - np.outer(Bv, lam)
        best = min(best, 0.5 * np.trace(E.T @ M @ E @ C))
    # P4: k = 3 space vs span{1, cos, sin}
    th = 2 * np.pi * np.arange(N) / N
    F = np.stack([np.ones(N), np.cos(th), np.sin(th)], axis=1)
    R3 = (Vt[:3] @ Chi).T
    Q1, _ = np.linalg.qr(R3); Q2, _ = np.linalg.qr(F)
    # sine of the largest principal angle, from the residual of projecting one space onto the other
    # (arccos of the cosines loses precision near 1: its floor is about sqrt(machine epsilon))
    ang = np.arcsin(min(1.0, np.linalg.svd(Q1 - Q2 @ (Q2.T @ Q1), compute_uv=False).max()))
    lam_n = np.real(np.fft.fft(C[0])) * np.real(np.fft.fft(Phi[0]))
    print(f"--- {label}")
    print(f"P1 circulant error  C: {circ_err(C):.1e}   Phi: {circ_err(Phi):.1e}")
    print(f"   chat*phihat by n = 0..5: {np.round(lam_n[:6] / lam_n[0], 4)}")
    print(f"   premise of part 3: top gap {simple:.3f}, top-vector entry sum {v.sum():.3f}")
    print(f"P2 k=1 read spread {spread:.1e}   write coefficient error (x N) {write_err:.1e}")
    print(f"P3 k=1 objective: Eckart-Young {obj_k1:.10g}  alternating LS {best:.10g}  rel diff {abs(best - obj_k1) / obj_k1:.1e}")
    print(f"P4 k=3 space vs span(1, cos, sin): largest principal angle {ang:.1e} rad")


analyse("exact symmetric model", *build())
analyse("control: H-read broken (random write columns)", *build(read_sym=False))
analyse("control: H-sym broken (sample not symmetrised)", *build(sym_inputs=False))
