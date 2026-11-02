# serves: T47
# serves: T48
# serves: T49
# serves: T50
# serves: T46d
"""Checks of Proposition T47, Theorem T48 and Proposition T49 (COARSE-THEORY.md B.R), from Definitions D43 to D45
(revision of referee report 22: class reads dq_c/dh_u, class signed read sum_c W~_cu q_c, an input where each
featured unit is off) written out here for dense ReLU networks and a finite or sampled base law.

P1  T47 part 1: pi(u) equals (e - 1/e)^2 / (2 (e^2 + e^-2 + (e + 1/e)^2)) (sympy, exact); lambda(u) > 0.
P2  T47 part 2: pi(v) = 0 and lambda(v) > 0 for the V circuit (car reader in layer 3).
P3  T47 part 3: pi(c) = 0 and lambda(c) > 0 for the ambulance; D42's eps_V > 0 on a ball around the inputs (sampled).
P4  T48: N_0 (L = 2, widths 3 and 3, n_y = 3) on X = [1, 2], uniform law (a fine grid): one route and lambda > 0 for
    every unit; 50 random perturbations of size 1e-3 stay perfectly clean.
P5  T49: the detector network (kappa 2, m 3), uniform law on its domain: lambda > 0 for every detector (quadrature).
P6  T50 in a finite instance: units relu(x_1), relu(x_2), never on together, read by one class (logits (h_1 + h_2, 0)),
    law uniform on (1, 0), (0, 1), (0, 0): xi(u_1) < 0 (exact, sympy).
P7  T46d: the same units read by the bias-free reader r = relu(h_1 + h_2): xi(u_1) = -P(u on) P(rest on) = -1/9 (exact).
Prediction (written before the run): P1 difference 0, lambda > 0; P2, P3 pi = 0, lambda > 0, D42 eps_V > 0; P4 all
perfectly clean; P5 all lambda > 0; P6 xi < 0; P7 xi = -1/9.
"""
import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(__file__))
from clean_eps import eps as eps42, sample_ball  # noqa: E402


def reduce_net(layers, W_out):
    out, scale = [], None
    for W, b in layers:
        W = W if scale is None else W * scale[None, :]
        n = np.linalg.norm(np.hstack([W, b[:, None]]), axis=1)
        out.append((W / n[:, None], b / n))
        scale = n
    return out, W_out * scale[None, :]


def forward(layers, W_out, c_out, x):
    hs, zs, h = [], [], x
    for W, b in layers:
        z = h @ W.T + b
        h = np.maximum(z, 0)
        zs.append(z); hs.append(h)
    y = h @ W_out.T + c_out
    q = np.exp(y - y.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
    return zs, hs, q


def indices(layers, W_out, c_out, x, nu, tol=1e-9):
    """pi and lambda of every live unit (Definition D45), base law nu on the rows of x (one position)."""
    layers, W_out = reduce_net(layers, W_out)
    zs, hs, q = forward(layers, W_out, c_out, x)
    L, res = len(layers), {}
    for l in range(L):
        below = x if l == 0 else hs[l - 1]
        for u in range(hs[l].shape[1]):
            h = hs[l][:, u]
            M = nu @ h
            if M <= tol:
                continue
            votes = below * layers[l][0][u][None, :]
            top = votes.max(1, keepdims=True)
            route = [tuple(np.where(votes[i] >= top[i] - tol)[0]) for i in range(len(x))]
            if l < L - 1:
                Wr = layers[l + 1][0][:, u]
                g = (zs[l + 1] > 0).astype(float) * Wr[None, :]
                S = g.sum(1)
                T = np.abs(Wr).sum()
            else:  # classes: g_c = dq_c/dh_u = q_c (W_cu - sum q W_.u); S = sum_c W~_cu q_c with W~ the centred readout
                Wr = W_out[:, u]
                g = q * (Wr[None, :] - (q @ Wr)[:, None])
                Wc = Wr - Wr.mean()
                S = q @ Wc
                T = np.abs(Wc).sum()
            mu = nu * h / M
            groups = {}
            for i, r in enumerate(route):
                if mu[i] > 0:
                    groups.setdefault(r, []).append(i)
            num = sum(mu[ix].sum() * (mu[ix] @ g[ix] / mu[ix].sum()) for ix in map(np.array, groups.values()))
            den = sum(mu[ix].sum() * np.linalg.norm(mu[ix] @ g[ix] / mu[ix].sum()) for ix in map(np.array, groups.values()))
            pi = 0.0 if den <= tol else 1 - float(num @ num) / den ** 2
            lam = 0.0 if T <= tol else float((mu @ S - nu @ S) / T)
            res[(l, u)] = dict(pi=pi, lam=lam, routes=len(groups))
    return res


I2 = np.eye(2)
x3pts = np.array([[1., 0.], [0., 1.], [0., 0.]])
nu3 = np.full(3, 1 / 3)
# P1: two jobs (inputs A, B, O)
L1 = [(I2.copy(), np.zeros(2)), (np.array([[1., 1.], [1., 0.], [0., 1.]]), np.zeros(3))]
Wo1 = np.array([[1., 0., -2.], [1., -2., 0.], [0., 0., 0.]])
r1 = indices(L1, Wo1, np.zeros(3), x3pts, nu3)
e = sp.E
Z = e + 1 / e + 1
gA = sp.Matrix([e, 1 / e, -(e + 1 / e)]) / Z ** 2
gB = sp.Matrix([1 / e, e, -(e + 1 / e)]) / Z ** 2
mean = (gA + gB) / 2
pi_sym = 1 - mean.dot(mean) / (sp.sqrt(gA.dot(gA)) / 2 + sp.sqrt(gB.dot(gB)) / 2) ** 2
target = (e - 1 / e) ** 2 / (2 * (e ** 2 + e ** -2 + (e + 1 / e) ** 2))
diff = sp.simplify(pi_sym - target)
print(f"P1: pi(u) numeric {r1[(1, 0)]['pi']:.6f}, closed form {float(target):.6f}, sympy difference {diff}; "
      f"routes {r1[(1, 0)]['routes']}; lambda(u) {r1[(1, 0)]['lam']:+.4f}")

# P2: the V case, with a car reader and a cat reader in layer 3 (inputs cat, car, O)
L2 = [(I2.copy(), np.zeros(2)), (np.array([[1., 1.], [1., 0.], [0., 1.]]), np.zeros(3)),
      (np.array([[1., 0., 2.], [0., 2., 0.]]), np.array([-2.5, -1.]))]
Wo2 = np.array([[0., 1.], [1., 0.], [0., 0.]])  # y_cat = K, y_car = R, y_0 = 0
r2 = indices(L2, Wo2, np.zeros(3), x3pts, nu3)
print(f"P2: pi(v) {r2[(1, 0)]['pi']:.2e} with {r2[(1, 0)]['routes']} routes; lambda(v) {r2[(1, 0)]['lam']:+.4f}")

# P3: the ambulance (four inputs and O)
L3 = [(np.eye(3), np.zeros(3)), (np.array([[1., 1., 0.], [0., 0., 1.]]), np.zeros(2))]
Wo3 = np.array([[1., -2.], [1., 2.], [0., 0.]])
c3 = np.array([0., -1., 0.])
x3 = np.array([[2., 1., 1.], [2., 1., 0.], [1., 2., 1.], [1., 2., 0.], [0., 0., 0.]])
r3 = indices(L3, Wo3, c3, x3, np.full(5, .2))
rng = np.random.default_rng(0)
xb3 = sample_ball([1.5, 1.5, 0.5], 1.5, 200000, rng)
e42_3 = eps42(L3, Wo3, xb3)["eps_V"]  # D42 has no readout bias; the bias does not enter its votes
print(f"P3: pi(c) {r3[(1, 0)]['pi']:.2e} with {r3[(1, 0)]['routes']} routes; lambda(c) {r3[(1, 0)]['lam']:+.4f}; "
      f"D42 eps_V {e42_3:.4f}")

# P4: Theorem T48, L = 2, widths (3, 3), n_y = 3
def n0(n=3, ny=3):
    s = 1 / (4 * n)
    W1 = np.ones((n, 1)); b1 = -np.arange(1, n + 1) / (8 * n)
    xs = np.linspace(1, 2, 4001)[:, None]
    h1 = np.maximum(xs @ W1.T + b1, 0)
    W2 = np.full((n, n), s); W2[:, 0] = 1
    base = h1 @ W2[0]
    lo, hi = base[0], base[-1]
    b2 = -(lo + (hi - lo) * (np.arange(1, n + 1) / (n + 1)))  # thresholds inside (1, 2), increasing in k
    Wo = np.zeros((ny, n)); Wo[0] = 1
    return [(W1, b1), (W2, b2)], Wo


xs = np.linspace(1, 2, 4001)[:, None]
nu = np.full(len(xs), 1 / len(xs))
lay0, Wo0 = n0()
r4 = indices(lay0, Wo0, np.zeros(3), xs, nu)
ok0 = all(v["routes"] == 1 and v["pi"] < 1e-12 and v["lam"] > 0 for v in r4.values())
print(f"P4: N_0 units {len(r4)}; all one route, pi = 0, lambda > 0: {ok0}; "
      f"smallest lambda {min(v['lam'] for v in r4.values()):.4f}")
okp = 0
for t in range(50):
    r_ = np.random.default_rng(100 + t)
    lay = [(W + 1e-3 * r_.standard_normal(W.shape), b + 1e-3 * r_.standard_normal(b.shape)) for W, b in lay0]
    Wo = Wo0 + 1e-3 * r_.standard_normal(Wo0.shape)
    rr = indices(lay, Wo, 1e-3 * r_.standard_normal(3), xs, nu)
    okp += all(v["routes"] == 1 and v["pi"] < 1e-9 and v["lam"] > 0 for v in rr.values()) and len(rr) == len(r4)
print(f"P4: perturbed copies perfectly clean: {okp}/50")

# P5: Proposition T49, detector network kappa = 2, m = 3, uniform law on X = {|y_1| + |y_2| <= 1} (blocks in R^2).
# By the block structure, an integral over X of a function of block 1 alone is the integral over block 1 of
# f(y) * pi (1 - |y|)^2. E_nu[q_1] - 1/2 = sum over caps of the integral of (sigma(c h) - 1/2), divided by vol(X).
import itertools as _it
from fractions import Fraction as _Fr
sys.path.insert(0, os.path.dirname(__file__))
from t22_reduction import circle_pt  # noqa: E402
kap, mm = 2, 3
xs_ = [np.array([float(a) for a in circle_pt(i)]) for i in range(1, mm + 1)]
eta = 1 - max(float(a @ b) for a, b in _it.permutations(xs_, 2))
rr = 1 - eta / 8
cc = 2 / (1 - rr)
sig = lambda t: 1 / (1 + np.exp(-t))
volX = np.pi ** 2 / 6
nr, nt = 3000, 3000
rad = np.linspace(rr, 1.0, nr + 1); rad = (rad[:-1] + rad[1:]) / 2; dr = (1 - rr) / nr
caps = []
for a in xs_:
    th0, hw = np.arctan2(a[1], a[0]), np.arccos(rr)
    th = np.linspace(th0 - hw, th0 + hw, nt + 1); th = (th[:-1] + th[1:]) / 2; dth = 2 * hw / nt
    R_, T_ = np.meshgrid(rad, th, indexing="ij")
    h = np.maximum(R_ * np.cos(T_ - th0) - rr, 0)
    w = np.pi * (1 - R_) ** 2 * R_ * dr * dth * (h > 0)
    caps.append(dict(V=w.sum(), A=(w * (sig(cc * h) - .5)).sum() / w.sum(),
                     Aw=(w * h * (sig(cc * h) - .5)).sum() / (w * h).sum()))
Enu = .5 + kap * sum(cp["V"] * cp["A"] for cp in caps) / volX
lams = [(cc / (2 * cc)) * ((.5 + cp["Aw"]) - Enu) for cp in caps]
print(f"P5: caps volume share {kap * sum(cp['V'] for cp in caps) / volX:.2e}; lambda of the detectors "
      f"{[f'{l:.4f}' for l in lams]}; all > 0: {all(l > 0 for l in lams)}")

# P6 and P7: the excess read in finite instances (sympy, exact)
pts = [(1, 0), (0, 1), (0, 0)]
nuw = [sp.Rational(1, 3)] * 3
h1 = lambda p: max(p[0], 0)
rest = lambda p: max(p[1], 0)  # the other unit, relu(x_2)
M = sum(w * h1(p) for w, p in zip(nuw, pts))
mu = [w * h1(p) / M for w, p in zip(nuw, pts)]
sgm = lambda t: 1 / (1 + sp.exp(-t))


def lam_pair(Sfun, T):
    """lambda and lambda_ind for a signed read S(t, R) with the unit's value t and the rest R."""
    lam = (sum(m_ * Sfun(h1(p), rest(p)) for m_, p in zip(mu, pts))
           - sum(w * Sfun(h1(p), rest(p)) for w, p in zip(nuw, pts))) / T
    lind = (sum(m_ * w2 * Sfun(h1(p), rest(p2)) for m_, p in zip(mu, pts) for w2, p2 in zip(nuw, pts))
            - sum(w * w2 * Sfun(h1(p), rest(p2)) for w, p in zip(nuw, pts) for w2, p2 in zip(nuw, pts))) / T
    return sp.simplify(lam), sp.simplify(lind)


# P6: class reader; logits (h_1 + h_2, 0); centred read of u_1 = (1/2, -1/2), so S = q_1 - 1/2 and T = 1
l6, li6 = lam_pair(lambda t, R: sgm(t + R) - sp.Rational(1, 2), 1)
xi6 = sp.simplify(l6 - li6)
print(f"P6: lambda {float(l6):.5f}, lambda_ind {float(li6):.5f}, xi {float(xi6):.5f}; xi < 0: {bool(xi6 < 0)}")
# P7: bias-free unit reader r = relu(h_1 + h_2): S = 1[t + R > 0], T = 1
l7, li7 = lam_pair(lambda t, R: 1 if t + R > 0 else 0, 1)
xi7 = sp.simplify(l7 - li7)
print(f"P7: lambda {l7}, lambda_ind {li7}, xi {xi7}; equals -1/9: {xi7 == sp.Rational(-1, 9)}")

