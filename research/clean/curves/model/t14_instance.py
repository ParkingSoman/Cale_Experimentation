# serves: T14
"""Exact instance of the construction in Theorem T14 (COARSE-THEORY.md B.6). It checks the reduction identity, not the
hardness (which is cited).

T: a 3 x 3 x 3 tensor with small integer entries. Chain: positions {0, 1, 2}, window Q = {0, 1, 2}, rho = 1 at 0,
gates 1, K = I_3, w_1(q) = T_q, H = [I_3; -I_3]. Law: uniform on the M = 16 rows of 9 non-constant columns of the
Sylvester Hadamard matrix of order 16.

Predictions (written before the run):
  H1  E u = 0 and Cov u = I exactly (integer arithmetic).
  H2  the chain's logits equal H sum_q T_q a(x, q) on every input (exact).
  H3  for 500 random builds of widths (1, 1), D2 = |T - h (x) kappa (x) omega|_F^2 + 1/2 |eta_perp|^2 |omega|^2 |kappa|^2
      (relative error about 1e-14), with P_perp eta = H h + eta_perp.
  H4  the best (1, 1) build found by 200 starts of alternating least squares has D2 = |T|_F^2 - sigma^2, where sigma is
      the spectral norm found by the same alternating method from 200 starts and checked against 10^6 random unit
      triples (no triple exceeds sigma by more than 1e-9 relative).
  H5  the visible width of layer 1 for k = 1 (T13) is larger than 1, so the width-(1, 1) build lies below it.
"""
import json
import numpy as np

rng = np.random.default_rng(14)
a = b = c = 3
T = rng.integers(-3, 4, size=(a, b, c)).astype(float)
Tq = [T[:, :, q] for q in range(c)]
H = np.concatenate([np.eye(a), -np.eye(a)])
ny = 2 * a
P = np.eye(ny) - np.ones((ny, ny)) / ny


def sylvester(m):
    Hm = np.array([[1]])
    while Hm.shape[0] < m:
        Hm = np.block([[Hm, Hm], [Hm, -Hm]])
    return Hm


M = 16
U = sylvester(M)[:, 1:1 + b * c].astype(float)          # rows = inputs; u(x) in R^{bc}
A = U.reshape(M, c, b)                                    # a(x, q) = A[x, q]
res = {"H1 mean max": float(np.abs(U.mean(0)).max()), "H1 cov - I max": float(np.abs(U.T @ U / M - np.eye(b * c)).max())}

# teacher through the chain: layer 1 v1(p) = a(x, p); layer 2 at p = 0: sum_q w(q) v1(0 + q); readout rho(0) = 1
v1 = A                                                    # (M, positions, b)
v2 = sum(np.einsum("ij,xj->xi", Tq[q], v1[:, q]) for q in range(c))
y = v2 @ H.T
res["H2 err"] = float(np.abs(y - np.einsum("yi,xi->xy", H, sum(A[:, q] @ Tq[q].T for q in range(c)))).max())


def d2_build(kappa, omega, eta, cb=np.zeros(ny)):
    yb = np.outer(sum(omega[q] * (A[:, q] @ kappa) for q in range(c)), eta) + cb
    return 0.5 * np.mean(np.sum(((yb - y) @ P) ** 2, axis=1))


errs = []
for _ in range(500):
    kappa, omega, eta = rng.normal(size=b), rng.normal(size=c), rng.normal(size=ny)
    pe = P @ eta
    h = H.T @ pe / 2
    eperp = pe - H @ h
    pred = np.sum((T - np.einsum("i,j,q->ijq", h, kappa, omega)) ** 2) + 0.5 * eperp @ eperp * (omega @ omega) * (kappa @ kappa)
    errs.append(abs(d2_build(kappa, omega, eta) - pred) / pred)
res["H3 max rel err"] = float(max(errs))


def hopm(T, x, y_, z, it=500):
    for _ in range(it):
        x = np.einsum("ijq,j,q->i", T, y_, z); x /= np.linalg.norm(x)
        y_ = np.einsum("ijq,i,q->j", T, x, z); y_ /= np.linalg.norm(y_)
        z = np.einsum("ijq,i,j->q", T, x, y_); z /= np.linalg.norm(z)
    return abs(np.einsum("ijq,i,j,q->", T, x, y_, z)), (x, y_, z)


best = (0, None)
for _ in range(200):
    v, vecs = hopm(T, *[rng.normal(size=n) for n in (a, b, c)])
    if v > best[0]:
        best = (v, vecs)
sigma, (hx, kx, ox) = best
X = rng.normal(size=(10 ** 6, a)); X /= np.linalg.norm(X, axis=1, keepdims=True)
Y = rng.normal(size=(10 ** 6, b)); Y /= np.linalg.norm(Y, axis=1, keepdims=True)
Zz = rng.normal(size=(10 ** 6, c)); Zz /= np.linalg.norm(Zz, axis=1, keepdims=True)
rand_max = float(np.abs(np.einsum("ijq,ni,nj,nq->n", T, X, Y, Zz)).max())
res["sigma (alternating, 200 starts)"] = float(sigma)
res["max over 10^6 random unit triples"] = rand_max
res["H4 random triple exceeds sigma"] = bool(rand_max > sigma * (1 + 1e-9))
# the build from the spectral triple: h = sigma x, kappa = y, omega = z, eta = H h (eta_perp = 0)
s_sign = np.sign(np.einsum("ijq,i,j,q->", T, hx, kx, ox))
d2_opt = d2_build(kx, ox, H @ (s_sign * sigma * hx))
res["|T|_F^2"] = float(np.sum(T ** 2))
res["D2 of the build from the spectral triple"] = float(d2_opt)
res["|T|^2 - sigma^2"] = float(np.sum(T ** 2) - sigma ** 2)
res["H4 rel err"] = float(abs(d2_opt - (np.sum(T ** 2) - sigma ** 2)) / (np.sum(T ** 2) - sigma ** 2))
# independent: polish random builds by gradient-free alternating least squares on (kappa, omega, eta)
vals = []
for _ in range(200):
    kappa, omega, h = rng.normal(size=b), rng.normal(size=c), rng.normal(size=a)
    for _ in range(300):
        kappa = np.einsum("ijq,i,q->j", T, h, omega) / ((h @ h) * (omega @ omega))
        omega = np.einsum("ijq,i,j->q", T, h, kappa) / ((h @ h) * (kappa @ kappa))
        h = np.einsum("ijq,j,q->i", T, kappa, omega) / ((kappa @ kappa) * (omega @ omega))
    vals.append(d2_build(kappa, omega, H @ h))
res["best ALS build D2 (200 starts)"] = float(min(vals))
res["H4 ALS vs |T|^2 - sigma^2 rel"] = float(abs(min(vals) - (np.sum(T ** 2) - sigma ** 2)) / (np.sum(T ** 2) - sigma ** 2))

# H5 visible width of layer 1 at k = 1
yc = (y - y.mean(0)) @ P
mu, E = np.linalg.eigh(yc.T @ yc / M); e1 = E[:, np.argmax(mu)]
reads = np.stack([e1 @ P @ H @ Tq[q] for q in range(c)])
res["H5 visible width d_1 (k = 1)"] = int(np.linalg.matrix_rank(reads, tol=1e-10))
res["floor 1/2 sum_{j>1} mu_j (T13 value at widths (d_1, 1))"] = float(0.5 * np.sum(np.sort(mu)[::-1][1:]))
res["T"] = T.astype(int).tolist()
print(json.dumps(res, indent=1))
json.dump(res, open(__file__.replace(".py", ".json"), "w"), indent=1)
