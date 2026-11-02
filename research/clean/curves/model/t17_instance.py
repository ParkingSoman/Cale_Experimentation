# serves: T17
# serves: T17.1
# serves: T17.2
# serves: T17.3
"""Exact instance for Part B.7 of COARSE-THEORY.md: formulas for the gates' reads.

Self-gated chain of depth 1, one read column, input field a(x, p) Gaussian and independent over 2 positions with
weights rho = (0.6, 0.4) (H-gauss). Moments are exact: for quadratic forms of a Gaussian vector, mean and covariance
have closed forms, so no sampling is used for the affine gate. The curved gate uses Gauss-Hermite quadrature (smooth
integrand, 3 dimensions).

Predictions (written before the run):
  V1  (T17 part 1, both linear-seen cases: centred law; law with mean and a constant entry, the constant direction
      kept) the residual y - y_B of the build B_V is orthogonal to 1, w_j and w_j w_l (exact moments, about 1e-10
      relative), and Cov(e, w) = 0 for e = a - Gamma w.
  V2  (T17 part 1) over 300 random reads q, with the best head and bias for each (exact least squares), and over a local
      minimisation of that value over q from 20 starts, the value is never below D2(B_V) (about 1e-9 relative), and
      the minimisation ends at D2(B_V).
  V3  (T17 part 3) if s lies in V: the effective read of B_V gives the network's gate argument, C_e s = 0.
  V4  (T17.2) centred law, k = 1: D2 of the T17 build at kernel Sigma^{-1/2} u (exact moments) equals
      (rho_2 / 2) (f_tot - f(u)) for random u; the best real root of the polynomial of part 3 equals the maximum of f
      found by 200000 random unit vectors plus local refinement (about 1e-9 relative); the number of real roots that
      are not eigenvalues is at most 3n - 2; the copy u = stilde / |stilde| and every top eigenvector of Q score
      strictly less.
  V5  (T17.3) centred law, k = 2: D2 of the T17 build equals (rho_2 / 2) (f(I) - f(Pi)) for random Pi; at the best of
      40 local maximisations over rank-2 projections, |Pi G - G Pi| is about 1e-6 relative to |G| or less, and every
      eigenvector of G(Pi) in the range of Pi lies in the span of the two resolvent vectors of part 3. Reported, not
      predicted: whether the fixed-point iteration Pi -> top-2 eigenvectors of G(Pi) reaches the best value
      (Conjecture D8).
  V7  (T17.3 part 4, added after referee report B7) at k = 2: for random rank-2 projections the formula
      f(Pi) = (alpha^2 |p|^2 + beta^2) tr(Pi_W' Q) + 2 alpha^2 p^T Q p + beta^2 p^T Q p / |p|^2 holds (about 1e-12), and the
      maximum of h(p) over the sphere |p|^2 = p^T stilde (random points plus local refinement) equals the best f of V5
      (about 1e-9 relative).
  V6  (T17.1) softplus gate phi(t) = log(1 + exp(t)) (L = 1), d = 3, one position, centred law: the seen-mean formula of
      part 1 leaves a residual orthogonal to tanh(w_1), w_1^3, w_1 w_2 and 1 (quadrature, about 1e-8); the identity
      of part 2 holds; the bound of part 3 holds; the price tends to zero as V turns toward s.
Revision after the first run (script errors, not results): (1) the V1 orthogonality metric used the residual at one
position without the constant alpha M C_e s that the bias adds once for all positions; the residual of one position is
delta(a_p) + alpha M C_e s. (2) The V4 brute force divided an unnormalised score by |u|^4, which is wrong for the
beta^2 term (degree 2); u is now normalised first.
"""
import json
import os

import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(17)
OUT = {}


def psd_sqrt(S, inv=False):
    e, U = np.linalg.eigh(S)
    e = np.clip(e, 0, None)
    if inv:
        e = np.where(e > 1e-12 * e.max(), 1 / np.sqrt(np.where(e > 0, e, 1)), 0.0)
    else:
        e = np.sqrt(e)
    return U @ np.diag(e) @ U.T


class Quad:
    """A vector of quadratics a^T Q_i a + l_i^T a + c_i in a ~ N(m, Sigma) at one position (Q_i symmetric)."""

    def __init__(self, Q, l, c):
        self.Q, self.l, self.c = np.asarray(Q, float), np.asarray(l, float), np.asarray(c, float)

    def __sub__(self, o):
        return Quad(self.Q - o.Q, self.l - o.l, self.c - o.c)

    def stack(self, o):
        return Quad(np.concatenate([self.Q, o.Q]), np.concatenate([self.l, o.l]), np.concatenate([self.c, o.c]))


def mean(q, m, S):
    return np.einsum("iab,ab->i", q.Q, S) + np.einsum("a,iab,b->i", m, q.Q, m) + q.l @ m + q.c


def covm(q, r, m, S):
    """Cov(q_i(a), r_j(a)) for a ~ N(m, S): 2 tr(Q_i S R_j S) + (2 Q_i m + l_i)^T S (2 R_j m + l'_j)."""
    t1 = 2 * np.einsum("iab,bc,jcd,da->ij", q.Q, S, r.Q, S)
    gq = 2 * np.einsum("iab,b->ia", q.Q, m) + q.l
    gr = 2 * np.einsum("iab,b->ia", r.Q, m) + r.l
    return t1 + gq @ S @ gr.T


def gated(read, N, al, be):
    """(al read^T a + be) N a as a Quad (one position, no constant)."""
    ny = N.shape[0]
    Q = np.stack([al * (np.outer(read, N[i]) + np.outer(N[i], read)) / 2 for i in range(ny)])
    return Quad(Q, be * N, np.zeros(ny))


def lin(L):
    """Linear features L a as a Quad."""
    return Quad(np.zeros((L.shape[0],) + (L.shape[1],) * 2), L, np.zeros(L.shape[0]))


def outer_feats(Kp):
    """All products w_j w_l, j <= l, of w = Kp^T a, as a Quad."""
    k = Kp.shape[1]
    Qs = [(np.outer(Kp[:, j], Kp[:, l]) + np.outer(Kp[:, l], Kp[:, j])) / 2 for j in range(k) for l in range(j, k)]
    return Quad(np.stack(Qs), np.zeros((len(Qs), Kp.shape[0])), np.zeros(len(Qs)))


rho = np.array([0.6, 0.4])
rho1, rho2 = rho.sum(), (rho ** 2).sum()


def d2_two_pos(delta1, const, P, m, S):
    """D2 for y_B - y = sum_p rho(p) delta1(a(x, p)) + const, positions independent with one law."""
    mu1 = mean(delta1, m, S)
    C1 = covm(delta1, delta1, m, S)
    tot_mean = rho1 * mu1 + const
    return 0.5 * (np.trace(P @ (rho2 * C1) @ P) + np.sum((P @ tot_mean) ** 2))


def build_case(centred):
    d0, N1, ny, k = 4, 3, 3, 2
    if centred:
        d = d0
        L = rng.normal(size=(d, d)); S = L @ L.T + 0.2 * np.eye(d); m = np.zeros(d)
    else:  # xi ~ N(m_xi, S_xi) in R^3 plus a constant entry 1 (last coordinate)
        d = d0
        L = rng.normal(size=(3, 3)); Sx = L @ L.T + 0.2 * np.eye(3)
        S = np.zeros((d, d)); S[:3, :3] = Sx
        m = np.concatenate([rng.normal(size=3), [1.0]])
    K = rng.normal(size=(d, N1)); s1 = rng.normal(size=N1); Hh = rng.normal(size=(ny, N1)); c = rng.normal(size=ny)
    al, be = 1.3, 0.6
    s = K @ s1; M = Hh @ K.T
    P = np.eye(ny) - np.ones((ny, ny)) / ny
    if centred:
        Kp = rng.normal(size=(d, k))
    else:  # keep the constant direction: one unit reads the constant entry alone
        Kp = np.zeros((d, k)); Kp[:, 0] = rng.normal(size=d); Kp[:, 0][-1] = 0.3; Kp[-1, 1] = 1.0
    return dict(d=d, k=k, ny=ny, S=S, m=m, s=s, M=M, c=c, al=al, be=be, P=P, Kp=Kp)


def t17_build(cs):
    S, m, Kp, s, M, al = cs["S"], cs["m"], cs["Kp"], cs["s"], cs["M"], cs["al"]
    M2 = S + np.outer(m, m)
    Gam = M2 @ Kp @ np.linalg.pinv(Kp.T @ M2 @ Kp)
    Ce = (np.eye(cs["d"]) - Gam @ Kp.T) @ S @ (np.eye(cs["d"]) - Gam @ Kp.T).T  # Cov(e); e has mean 0 if linear-seen
    q = Gam.T @ s
    Hp = M @ Gam
    cp = cs["c"] + al * rho1 * M @ Ce @ s
    return Gam, Ce, q, Hp, cp


def v1_v3(centred):
    cs = build_case(centred)
    S, m, Kp, s, M, al, be, P, c = (cs[x] for x in ("S", "m", "Kp", "s", "M", "al", "be", "P", "c"))
    Gam, Ce, q, Hp, cp = t17_build(cs)
    e_mean = (np.eye(cs["d"]) - Gam @ Kp.T) @ m
    cov_ew = (np.eye(cs["d"]) - Gam @ Kp.T) @ S @ Kp
    teacher = gated(s, M, al, be)
    student = gated(Kp @ q, Hp @ Kp.T, al, be)
    delta = student - teacher                          # per position; constant part: c' - c
    const = cp - c
    D2B = d2_two_pos(delta, const, P, m, S)
    # orthogonality of the residual (per position, since positions are independent and the residual at p is centred
    # given w(x, p)) to 1, w_j, w_j w_l
    h = lin(Kp.T).stack(outer_feats(Kp))
    resid_mean = mean(delta, m, S) * rho1 + const       # E (y_B - y)
    E_dh = covm(delta, h, m, S) + np.outer(mean(delta, m, S), mean(h, m, S))  # E[delta1 h] at one position
    # E[(y - m_V) h(w_p)] = rho(p) E[delta1(a_p) h(w_p)] + E[rest] E h with rest = other position and constant, mean 0
    scale = np.abs(covm(teacher, teacher, m, S)).max()
    res = {
        "D2(B_V)": D2B,
        "max |E e| (linear-seen => 0)": float(np.abs(e_mean).max()),
        "max |Cov(e, w)|": float(np.abs(cov_ew).max()),
        "max |E (y_B - y)|": float(np.abs(P @ resid_mean).max()),
        "max |Cov(residual_p, h(w_p))| / scale": float(np.abs(P @ covm(delta, h, m, S)).max() / scale),
        "max |E residual_p| / scale": float(np.abs(P @ (mean(delta, m, S) + al * M @ Ce @ s)).max() / scale),
    }
    # V2: random reads with the best head and bias, and local minimisation over q
    Py_t = teacher

    def best_for_read(qq):
        f = gated(Kp @ qq, Kp.T, al, be)               # features (al qq^T w + be) w, k of them
        Cf = rho2 * covm(f, f, m, S)
        Cyf = rho2 * covm(Py_t, f, m, S)
        Cyy = rho2 * covm(Py_t, Py_t, m, S)
        Cf_inv = np.linalg.pinv(Cf)
        return 0.5 * np.trace(P @ (Cyy - Cyf @ Cf_inv @ Cyf.T) @ P)

    vals = [best_for_read(rng.normal(size=cs["k"]) * 3) for _ in range(300)]
    loc = [minimize(best_for_read, rng.normal(size=cs["k"]), method="Nelder-Mead",
                    options=dict(xatol=1e-12, fatol=1e-14, maxiter=4000)).fun for _ in range(20)]
    res["value at q = Gamma^T s with best head (should equal D2(B_V))"] = best_for_read(q)
    res["min over 300 random reads"] = float(min(vals))
    res["min over 20 local minimisations"] = float(min(loc))
    res["V2 max (D2(B_V) - value)/D2(B_V) (<= ~0 expected)"] = float(max(D2B - min(vals), D2B - min(loc)) / D2B)
    # V3: V containing s
    if centred:
        cs3 = dict(cs); Kp3 = np.stack([s, rng.normal(size=cs["d"])], 1); cs3["Kp"] = Kp3
        G3, Ce3, q3, _, _ = t17_build(cs3)
        res["V3 |C_e s| with s in V"] = float(np.abs(Ce3 @ s).max())
        res["V3 |(K' q') - s| (read equals s)"] = float(np.abs(Kp3 @ q3 - s).max())
    return res


OUT["V1-V3 centred"] = v1_v3(True)
OUT["V1-V2 mean + constant entry"] = v1_v3(False)


# ---------- V4, V5: centred law, choice of V ----------
def case_centred(d=4, ny=3):
    L = rng.normal(size=(d, d)); S = L @ L.T + 0.2 * np.eye(d)
    K = rng.normal(size=(d, 3)); s = K @ rng.normal(size=3); M = rng.normal(size=(ny, 3)) @ K.T
    P = np.eye(ny) - np.ones((ny, ny)) / ny
    Sh = psd_sqrt(S)
    st = Sh @ s; Bm = P @ M @ Sh; Q = Bm.T @ Bm
    return dict(d=d, ny=ny, S=S, m=np.zeros(d), s=s, M=M, c=rng.normal(size=ny), P=P, Sh=Sh, st=st, Q=Q,
                al=1.1, be=0.7)


cc = case_centred()
al, be, st, Q = cc["al"], cc["be"], cc["st"], cc["Q"]


def f_pi(Pi):
    return al ** 2 * ((st @ Pi @ st) * np.trace(Pi @ Q) + st @ Pi @ Q @ Pi @ st) + be ** 2 * np.trace(Pi @ Q)


def d2_of_kernel(Kp):
    cs = dict(cc); cs["Kp"] = Kp; cs["k"] = Kp.shape[1]
    _, _, q, Hp, cp = t17_build(cs)
    delta = gated(Kp @ q, Hp @ Kp.T, al, be) - gated(cc["s"], cc["M"], al, be)
    return d2_two_pos(delta, cp - cc["c"], cc["P"], cc["m"], cc["S"])


Shi = np.linalg.inv(cc["Sh"])
v4 = {}
errs = []
for _ in range(20):
    u = rng.normal(size=cc["d"]); u /= np.linalg.norm(u)
    errs.append(abs(d2_of_kernel((Shi @ u)[:, None]) - rho2 / 2 * (f_pi(np.eye(cc["d"])) - f_pi(np.outer(u, u))))
                / d2_of_kernel((Shi @ u)[:, None]))
v4["max rel err D2 vs (rho_2/2)(f_tot - f(u))"] = float(max(errs))
fu = lambda u: f_pi(np.outer(u, u) / (u @ u))
X = rng.normal(size=(200000, cc["d"])); X /= np.linalg.norm(X, axis=1)[:, None]
vals = (2 * al ** 2 * (X @ st) ** 2 + be ** 2) * np.einsum("ij,jk,ik->i", X, Q, X)
starts = X[np.argsort(vals)[-20:]]
brute = max(-minimize(lambda u: -fu(u), x0, method="BFGS", options=dict(gtol=1e-12)).fun for x0 in starts)
lam_all, E = np.linalg.eigh(Q)
# distinct eigenvalues and r_j
lam, rr = [], []
for lv, vec in zip(lam_all, E.T):
    if lam and abs(lv - lam[-1]) < 1e-9 * max(1, abs(lv)):
        rr[-1] += (vec @ st) ** 2
    else:
        lam.append(lv); rr.append((vec @ st) ** 2)
lam, rr = np.array(lam), np.array(rr)
n = len(lam)
pm = np.polynomial.Polynomial
prod = pm([1.0])
for l in lam:
    prod = prod * pm([-l, 1.0])
Ap = sum(r * (prod // pm([-l, 1.0])) for r, l in zip(rr, lam))                       # A prod
Dp = sum(r * (prod // pm([-l, 1.0])) ** 2 for r, l in zip(rr, lam))                  # D prod^2
Bp = sum(l * r * (prod // pm([-l, 1.0])) ** 2 for r, l in zip(rr, lam))              # B prod^2
gp = 2 * al ** 2 * Ap * (Ap * prod - Bp) + be ** 2 * Dp * prod                       # g prod^3
roots = [z.real for z in gp.roots() if abs(z.imag) < 1e-7 * max(1, abs(z))]
roots = [z for z in roots if np.min(np.abs(z - lam)) > 1e-9 * max(1, abs(z))]
cands = []
for z in roots:
    x = E @ np.linalg.solve(np.diag(z - lam_all), E.T @ st); x /= np.linalg.norm(x)
    cands.append((fu(x), z))
best_root = max(cands)
v4.update({
    "n distinct eigenvalues": n, "polynomial degree": gp.degree(), "3n - 2": 3 * n - 2,
    "real roots not eigenvalues": len(roots),
    "best root value": best_root[0], "best mu": best_root[1], "eigenvalues of Q": lam.tolist(),
    "brute-force max of f": brute,
    "rel gap root vs brute": abs(best_root[0] - brute) / brute,
    "copy f(stilde/|stilde|)": fu(st / np.linalg.norm(st)),
    "top eigenvector f": fu(E[:, -1]),
})
OUT["V4 width 1"] = v4

# V5
def proj_from(Xm):
    U, _ = np.linalg.qr(Xm.reshape(cc["d"], 2)); return U @ U.T


def G_of(Pi):
    S_ = np.outer(st, st)
    return al ** 2 * (np.trace(Pi @ Q) * S_ + (st @ Pi @ st) * Q + Q @ Pi @ S_ + S_ @ Pi @ Q) + be ** 2 * Q


v5 = {}
errs = []
for _ in range(20):
    U, _ = np.linalg.qr(rng.normal(size=(cc["d"], 2)))
    a_ = d2_of_kernel(Shi @ U)
    errs.append(abs(a_ - rho2 / 2 * (f_pi(np.eye(cc["d"])) - f_pi(U @ U.T))) / a_)
v5["max rel err D2 vs (rho_2/2)(f(I) - f(Pi))"] = float(max(errs))
best = None
for _ in range(40):
    r_ = minimize(lambda x: -f_pi(proj_from(x)), rng.normal(size=2 * cc["d"]), method="BFGS",
                  options=dict(gtol=1e-12))
    if best is None or r_.fun < best.fun:
        best = r_
Pi = proj_from(best.x)
Gm = G_of(Pi)
v5["best f over rank-2 projections"] = f_pi(Pi)
v5["|Pi G - G Pi| / |G|"] = float(np.linalg.norm(Pi @ Gm - Gm @ Pi) / np.linalg.norm(Gm))
gam = be ** 2 + al ** 2 * st @ Pi @ st
v = Q @ Pi @ st
th, Ev = np.linalg.eigh(Gm)
span_err = []
for t_, x in zip(th, Ev.T):
    if x @ Pi @ x > 0.5:  # eigenvector inside the range of Pi
        Rinv = np.linalg.inv(t_ / gam * np.eye(cc["d"]) - Q)
        Bas = np.stack([Rinv @ st, Rinv @ v], 1)
        coef = np.linalg.lstsq(Bas, x, rcond=None)[0]
        span_err.append(float(np.linalg.norm(Bas @ coef - x)))
v5["max distance of kept eigenvectors from the resolvent span"] = max(span_err)
# fixed-point iteration (report only)
scf = []
for _ in range(10):
    P_ = proj_from(rng.normal(size=2 * cc["d"]))
    for it in range(500):
        th_, Ev_ = np.linalg.eigh(G_of(P_))
        P_new = Ev_[:, -2:] @ Ev_[:, -2:].T
        if np.linalg.norm(P_new - P_) < 1e-13:
            break
        P_ = P_new
    scf.append(f_pi(P_))
v5["fixed-point iteration values from 10 starts (report only)"] = scf
v5["T17.2 width-1 best, for comparison"] = best_root[0]
OUT["V5 width 2"] = v5

# ---------- V7: T17.3 part 4, reduction to one vector p ----------
def compress_top(Q_, excl, m_):
    """Sum of the m_ largest eigenvalues of Q_ compressed to the orthogonal complement of the columns of excl."""
    Bq, _ = np.linalg.qr(excl)
    Pc = np.eye(Q_.shape[0]) - Bq @ Bq.T
    Uc = np.linalg.svd(Pc)[0][:, :Q_.shape[0] - Bq.shape[1]]           # orthonormal basis of the complement
    ev = np.linalg.eigvalsh(Uc.T @ Q_ @ Uc)
    return float(np.sort(ev)[::-1][:m_].sum())


v7 = {}
errs = []
for _ in range(20):
    Pi_r = proj_from(rng.normal(size=2 * cc["d"]))
    p_ = Pi_r @ st
    Wq = Pi_r - np.outer(p_, p_) / (p_ @ p_)
    rhs = (al ** 2 * p_ @ p_ + be ** 2) * np.trace(Wq @ Q) + 2 * al ** 2 * p_ @ Q @ p_ + be ** 2 * p_ @ Q @ p_ / (p_ @ p_)
    errs.append(abs(f_pi(Pi_r) - rhs) / abs(f_pi(Pi_r)))
v7["max rel err of the p-formula"] = float(max(errs))
rs_ = np.linalg.norm(st)


def h_of_z(zv):
    zv = zv / np.linalg.norm(zv)
    p_ = (st + rs_ * zv) / 2                      # the sphere |p|^2 = p^T stilde: centre stilde / 2, radius |stilde| / 2
    if p_ @ p_ < 1e-14:
        return -np.inf
    kap = compress_top(Q, np.stack([st, p_], 1), 1)
    return (al ** 2 * p_ @ p_ + be ** 2) * kap + 2 * al ** 2 * p_ @ Q @ p_ + be ** 2 * p_ @ Q @ p_ / (p_ @ p_)


Z = rng.normal(size=(4000, cc["d"]))
hv = np.array([h_of_z(zv) for zv in Z])
best_h = max(-minimize(lambda zv: -h_of_z(zv), Z[i], method="Nelder-Mead",
                       options=dict(xatol=1e-12, fatol=1e-12, maxiter=20000)).fun for i in np.argsort(hv)[-10:])
p0_val = be ** 2 * compress_top(Q, st[:, None], 2)
v7["max of h over the sphere"] = best_h
v7["value at p = 0"] = p0_val
v7["best f of V5"] = f_pi(Pi)
v7["rel gap max(h, p0) vs best f"] = abs(max(best_h, p0_val) - f_pi(Pi)) / f_pi(Pi)
OUT["V7 one-vector reduction"] = v7

# ---------- V6: curved gate (softplus), d = 3, one position, centred ----------
d = 3
L6 = rng.normal(size=(d, d)); S6 = L6 @ L6.T + 0.3 * np.eye(d); Sh6 = psd_sqrt(S6)
s6 = rng.normal(size=d); M6 = rng.normal(size=(3, d)); P6 = np.eye(3) - np.ones((3, 3)) / 3
phi = lambda t: np.logaddexp(0, t)
dphi = lambda t: 1 / (1 + np.exp(-t))
xg, wg = np.polynomial.hermite_e.hermegauss(60); wg = wg / wg.sum()            # standard normal nodes
G3 = np.stack(np.meshgrid(xg, xg, xg, indexing="ij"), -1).reshape(-1, 3)
W3 = np.einsum("i,j,k->ijk", wg, wg, wg).reshape(-1)
A6 = G3 @ Sh6.T                                                                 # a = Sigma^{1/2} g
y6 = phi(A6 @ s6)[:, None] * (A6 @ M6.T)


def E(f):
    return np.tensordot(W3, f, axes=(0, 0))


def v6_for(Kp):
    Gam = S6 @ Kp @ np.linalg.inv(Kp.T @ S6 @ Kp)
    Ce = (np.eye(d) - Gam @ Kp.T) @ S6 @ (np.eye(d) - Gam @ Kp.T).T
    sig = np.sqrt(max(s6 @ Ce @ s6, 0))
    w = A6 @ Kp
    t = w @ (Gam.T @ s6)
    ps = (phi(t[:, None] + sig * xg[None, :]) * wg).sum(1)
    pss = (dphi(t[:, None] + sig * xg[None, :]) * wg).sum(1)
    mV = ps[:, None] * (w @ (M6 @ Gam).T) + pss[:, None] * (M6 @ Ce @ s6)[None, :]
    yB = phi(t)[:, None] * (w @ (M6 @ Gam).T)
    r = (y6 - mV) @ P6
    hs = [np.ones(len(w)), np.tanh(w[:, 0]), w[:, 0] ** 3, w[:, 0] * w[:, 1]]
    orth = max(float(np.abs(E(r * h[:, None])).max()) for h in hs) / float(np.abs(E((y6 @ P6) ** 2)).max())
    F = 0.5 * E(((y6 - mV) @ P6) ** 2).sum()
    dlt = (yB - mV) @ P6
    dlt_c = dlt - E(dlt)
    rhs = F + 0.5 * E(dlt_c ** 2).sum()
    yBc = yB @ P6 - E(yB @ P6) + E(y6 @ P6)                                       # best bias
    D2B = 0.5 * E((yBc - y6 @ P6) ** 2).sum()
    Ce_h = psd_sqrt(Ce)
    bound = 1.0 * sig ** 2 * ((2 / np.pi) * E(((w @ (M6 @ Gam).T) @ P6) ** 2).sum()
                              + np.linalg.norm(P6 @ M6 @ Ce_h, 2) ** 2)
    return dict(sigma=float(sig), orth=orth, F=float(F), D2B=float(D2B), identity_err=float(abs(D2B - rhs) / D2B),
                price=float(D2B - F), bound=float(bound))


v6 = {}
other = rng.normal(size=d)
for ang in (1.2, 0.6, 0.3, 0.1, 0.0):
    # kept space spanned by "other" and a vector that turns toward s as ang -> 0
    u1 = np.cos(ang) * s6 / np.linalg.norm(s6) + np.sin(ang) * np.cross(s6, other) / np.linalg.norm(np.cross(s6, other))
    v6[f"angle {ang}"] = v6_for(np.stack([u1, other], 1))
OUT["V6 softplus"] = v6

print(json.dumps(OUT, indent=1, default=float))
json.dump(OUT, open(os.path.join(os.path.dirname(__file__), "t17_instance.json"), "w"), indent=1, default=float)
