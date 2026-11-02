# serves: T1b
"""Exact plane-model instance for Lemma T1b and Remark B1.1 (COARSE-THEORY.md B.1).

Positions: a polar grid (J rings x A angular bins), with A = N m, so the rotation gamma by 2 pi / N is an exact shift
of the bins by m. Inputs: fields x[o, j, a] with O = N m_o orientation channels; gamma shifts bins by m and
orientation channels by m_o (a rotation of the image rotates both where content is and which way it points).
Members (H-clean, exact): kernel K(i, p) = gamma^i . K(0, gamma^-i p), so h_{i+1}(gamma x, gamma p) = h_i(x, p).
Readers (H-read): r = N q channels, R = cyclic shift by q (a permutation of order N), W_{., i+1} = R W_{., i}.
Gates: reader pre-activation from the family itself, u(x, p) = W h(x, p) + bias (bias constant on reader orbits),
so the gates are equivariant by construction. Pooling: identity (commutes with gamma).
Law: a sample symmetrised over the N rotations (H-sym exact for the empirical law), then optionally an orientation
bias of strength delta (x[o] *= 1 + delta cos(2 angle_o)), which breaks H-sym for delta > 0.

Predictions (written before the run):
  P1 h_{i+1}(gamma x, gamma p) = h_i(x, p) to ~1e-15.
  P2 (Pi (x) R) T (Pi (x) R)^T = T for the GATED metric (no H-kron needed for this step) to ~1e-13 relative.
  P3 constant metric M (x) K: C and Phi circulant; k = 1 read constant, write = mean column (T1a parts 1-3).
  P4 gated metric (Remark B1.1): Lambda_{R j} = Pi Lambda_j for every reader j, to ~1e-12.
  P5 delta > 0: the errors of P2-P4 grow linearly in delta (log-log slope ~1 for small delta).
"""
import numpy as np

rng = np.random.default_rng(1)
N, m, m_o, J, q, S = 10, 2, 2, 3, 3, 300
A, O, r = N * m, N * m_o, N * q
P = J * A


def rot(x, k=1):
    """gamma^k on fields (..., O, J, A): shift orientation channels by k m_o and bins by k m."""
    return np.roll(np.roll(x, k * m_o, axis=-3), k * m, axis=-1)


def pos_rot(p, k=1):
    j, a = divmod(p, A)
    return j * A + (a + k * m) % A


# template kernels K(0, p): local, non-negative, orientation-tuned, different per position
ang_o = 2 * np.pi * np.arange(O) / O
K0 = np.zeros((P, O, J, A))
for p in range(P):
    j, a = divmod(p, A)
    tune = np.exp(np.cos(ang_o - 2 * np.pi * a / A) * 2.0)          # prefers the tangent direction at bin a
    for dj in (-1, 0, 1):
        jj = j + dj
        if 0 <= jj < J:
            for da in (-1, 0, 1):
                K0[p, :, jj, (a + da) % A] = tune * rng.uniform(0.5, 1.5) * np.exp(-(da ** 2 + dj ** 2))
Kmem = np.zeros((N, P, O, J, A))
for i in range(N):
    for p in range(P):
        Kmem[i, p] = rot(K0[pos_rot(p, -i)], i)                      # K(i, p) = gamma^i K(0, gamma^-i p)
b = -1.0
W0 = rng.normal(size=r)
Wcol = np.stack([np.roll(W0, i * q) for i in range(N)], axis=1)      # (r, N), column i = R^i W0
bias_r = np.tile(rng.normal(size=q) * 0.5, N)                         # constant on reader orbits (R shifts by q)
Mq = rng.normal(size=(r, r)); Mq = Mq @ Mq.T
Rm = np.roll(np.eye(r), q, axis=0)
Mc = sum(np.linalg.matrix_power(Rm, k) @ Mq @ np.linalg.matrix_power(Rm, k).T for k in range(N)) / N + np.eye(r)
Kpos = np.exp(-np.abs(np.subtract.outer(np.arange(P) // A, np.arange(P) // A))) * 0  # placeholder (local metric below)


def members(X):
    return np.maximum(np.einsum("sojk,ipojk->sip", X.reshape(len(X), O, J, A), Kmem) + b, 0.0)


def sample(delta):
    X = np.abs(rng.normal(size=(S, O, J, A))) ** 2
    X = np.concatenate([rot(X, k) for k in range(N)], axis=0)        # symmetrise over the group
    return X * (1 + delta * np.cos(2 * ang_o))[None, :, None, None]


def analyse(delta, X):
    H = members(X)                                                     # (s, N, P)
    U = np.einsum("ci,sip->scp", Wcol, H) + bias_r[None, :, None]
    G = (U > 0).astype(float)                                         # reader gates (s, r, P)
    out = {}
    if delta == 0:
        x0 = X[:5]
        lhs = members(rot(x0))                                         # h(gamma x, .)
        e1 = 0.0
        for i in range(N):
            for p in range(P):
                e1 = max(e1, np.abs(lhs[:, (i + 1) % N, pos_rot(p)] - members(x0)[:, i, p]).max())
        out["P1 identity error"] = e1
    # P2: T for the gated, position-local metric: T[(i,c),(j,d)] = E sum_p h_i h_j g_c delta_cd
    T = np.einsum("sip,sjp,scp->icj", H, H, G) / len(H)               # (N, r, N), diagonal in readers
    Tfull = np.zeros((N * r, N * r))
    for c in range(r):
        idx = np.arange(N) * r + c
        Tfull[np.ix_(idx, idx)] = T[:, c, :]
    Pi = np.roll(np.eye(N), 1, axis=0)
    Q = np.kron(Pi, Rm)
    out["P2 T invariance (rel)"] = np.abs(Q @ Tfull @ Q.T - Tfull).max() / np.abs(Tfull).max()
    # P3: constant Kronecker metric with K = identity on positions: C = E sum_p h h^T
    C = np.einsum("sip,sjp->ij", H, H) / len(H)
    Phi = Wcol.T @ Mc @ Wcol
    circ = lambda Z: np.abs(np.roll(np.roll(Z, 1, 0), 1, 1) - Z).max() / np.abs(Z).max()
    out["P3 C circulant"] = circ(C)
    out["P3 Phi circulant"] = circ(Phi)
    w, V = np.linalg.eigh(C); Ch = (V * np.sqrt(w)) @ V.T; Chi = (V / np.sqrt(w)) @ V.T
    ev, EV = np.linalg.eigh(Ch @ Phi @ Ch)
    read = EV[:, -1] @ Chi; read = read / read.mean()
    out["P3 k=1 read spread"] = np.abs(read - 1).max()
    lw = C @ np.ones(N) / (np.ones(N) @ C @ np.ones(N))
    out["P3 write coeff error (x N)"] = np.abs(lw - 1 / N).max() * N
    # P4: gated per-reader write vectors for the summed read t = sum_i h_i (T0 part 4)
    t = H.sum(1)                                                       # (s, P)
    Lam = np.einsum("scp,sip,sp->ci", G, H, t) / np.einsum("scp,sp->c", G, t ** 2)[:, None]   # (r, N)
    # substitution x -> gamma x, p -> gamma p gives Lam[c, i] = Lam[c - q, i - 1]: rolling Lam by q readers and by
    # one member must return Lam itself
    out["P4 Lambda_{Rj} - Pi Lambda_j"] = np.abs(np.roll(np.roll(Lam, q, axis=0), 1, axis=1) - Lam).max() / np.abs(Lam).max()
    return out


res = {}
for delta in (0.0, 0.01, 0.03, 0.1, 0.3):
    res[delta] = analyse(delta, sample(delta))
for k in res[0.0]:
    print(f"{k:32s} delta=0: {res[0.0][k]:.1e}")
print("P5 growth with delta (errors at delta = 0.01, 0.03, 0.1, 0.3; log-log slope from 0.01 to 0.1):")
for k in ("P2 T invariance (rel)", "P3 C circulant", "P3 k=1 read spread", "P3 write coeff error (x N)", "P4 Lambda_{Rj} - Pi Lambda_j"):
    v = [res[d][k] for d in (0.01, 0.03, 0.1, 0.3)]
    slope = np.log(v[2] / v[0]) / np.log(10)
    print(f"  {k:32s} {' '.join(f'{x:.1e}' for x in v)}   slope {slope:.2f}")
