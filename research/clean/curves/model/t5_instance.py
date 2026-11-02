# serves: T5
"""Exact two-family matrix instance for Theorem T5 (COARSE-THEORY.md B.2').

Two families (F = 2) of N = 10 members each, both shifts of their own template under the same Z_10 action on the
inputs; readers in orbits of one permutation R; a shift-invariant constant metric; the sample symmetrised over the
group.

Predictions (written before the run):
  P1 every N x N block of C and Phi is circulant (~1e-13 relative).
  P2 in the Fourier basis (index ordered by mode, then family) C^{1/2} Phi C^{1/2} is block-diagonal with 2 x 2
     blocks: off-block mass ~1e-13 of the total.
  P4 the reads themselves (eigenvectors mapped by C^{-1/2}) are also single-frequency (weight >= 1 - 1e-12), but
     mix the two families within a frequency (the map Chat(n)^{-1/2} depends on the data; T5 part 5).
  P3 each of the top 6 eigenvectors of C^{1/2} Phi C^{1/2} has all its weight on one frequency (n, -n) in both families
     (largest weight fraction >= 1 - 1e-12).
"""
import numpy as np

rng = np.random.default_rng(2)
N, m, q, S, F = 10, 6, 4, 3000, 2


def shift(a, k, axis=0):
    return np.roll(a, k, axis=axis)


X = np.abs(rng.normal(size=(S, N, m))) + 0.3 * rng.normal(size=(S, N, m))
X = np.concatenate([shift(X, k, axis=1) for k in range(N)], axis=0)
Hs = []
for f in range(F):
    K0 = np.abs(rng.normal(size=(N, m))) * np.exp(-np.minimum(np.arange(N), N - np.arange(N)) ** 2 / (3.0 + 3 * f))[:, None]
    Kmem = np.stack([shift(K0, i) for i in range(N)])
    Hs.append(np.maximum(np.einsum("snm,inm->si", X, Kmem) - 0.5, 0.0))
H = np.concatenate(Hs, axis=1)                                     # (s, F N), index f N + i
Ws = []
for f in range(F):
    W0 = rng.normal(size=(N, q))
    Ws.append(np.stack([shift(W0, i).ravel() for i in range(N)], axis=1))
W = np.concatenate(Ws, axis=1)                                     # (N q, F N)
A = rng.normal(size=(N * q, N * q)); A = A @ A.T
Pq = lambda k: np.kron(np.roll(np.eye(N), k, axis=0), np.eye(q))
M = sum(Pq(k) @ A @ Pq(k).T for k in range(N)) / N + np.eye(N * q)

C = H.T @ H / len(H)
Phi = W.T @ M @ W
blk = lambda Z, f, g: Z[f * N:(f + 1) * N, g * N:(g + 1) * N]
circ = lambda Z: np.abs(np.roll(np.roll(Z, 1, 0), 1, 1) - Z).max() / np.abs(Z).max()
print("P1 block circulant errors:", " ".join(f"{circ(blk(Z, f, g)):.1e}" for Z in (C, Phi) for f in range(F) for g in range(F)))

w, V = np.linalg.eigh(C); Ch = (V * np.sqrt(w)) @ V.T
Op = Ch @ Phi @ Ch
U = np.exp(-2j * np.pi * np.outer(np.arange(N), np.arange(N)) / N) / np.sqrt(N)
UF = np.kron(np.eye(F), U)
Oh = UF @ Op @ UF.conj().T
order = [f * N + n for n in range(N) for f in range(F)]            # (mode, family)
Oh = Oh[np.ix_(order, order)]
mask = np.kron(np.eye(N), np.ones((F, F))).astype(bool)
print(f"P2 off-block mass fraction {np.abs(Oh[~mask]).sum() / np.abs(Oh).sum():.1e}")

ev, EV = np.linalg.eigh(Op)
for k in range(1, 7):
    v = UF @ EV[:, -k]
    p = np.abs(v.reshape(F, N)) ** 2
    per_freq = np.array([p[:, n].sum() + (p[:, (N - n) % N].sum() if n not in (0, N // 2) else 0) for n in range(N // 2 + 1)])
    print(f"P3 eigenvector {k}: eigenvalue {ev[-k]:.4g}, frequency {per_freq.argmax()}, weight on it {per_freq.max():.15f}")

Chi = (V / np.sqrt(w)) @ V.T
for k in range(1, 7):
    rd = UF @ (Chi @ EV[:, -k])
    p = np.abs(rd.reshape(F, N)) ** 2
    per_freq = np.array([p[:, n].sum() + (p[:, (N - n) % N].sum() if n not in (0, N // 2) else 0) for n in range(N // 2 + 1)])
    fam = p.sum(1) / p.sum()
    print(f"P4 read {k}: frequency {per_freq.argmax()}, weight on it {per_freq.max() / per_freq.sum():.15f}, family shares {np.round(fam, 3)}")
