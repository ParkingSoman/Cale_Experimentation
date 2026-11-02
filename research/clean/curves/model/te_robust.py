# serves: TE4
"""Exact checks of Part B.E of COARSE-THEORY.md (Lemma TE0, Proposition TE1, Proposition TE3, Theorem TE4,
Corollary TE4.1).

A  Lemma TE0 on random (q, y, d): |KL(q||s(y+d)) - KL(q||s(y))| <= osc(d).
B  Proposition TE1 on TM4 part 2's instance: V_b(N_delta) = V_b(N^+) for every delta > 0, the drop does not depend on
   delta, and a near-least build's swing on j1 is at least (V_b(N_0) - V_b(N_delta) - eta) / (delta sum p g).
C  Corollary TE4.1 part 2's instance: V_rob = L_T for every delta (inf over heads of the worst case over U, exact:
   per point the worst case is at a vertex, TE3 part 3), and V_b = 0.
D  Corollary TE4.1 part 1's instance: the copy box gives worst-case loss L_T on U; a multistart search over boxes and
   heads finds nothing below L_T (beyond solver tolerance); V_b = 0 via the 1/delta box.
F  Proposition TE5 on the referee's instance: V_rob lies between the proof's lower bound and its limit, and B^0's
   loss grows without bound.
E  (in te_robust_e.py) Theorem TE4 on random instances with three logits, two grouped regions (one group at size 0, one with a fixed
   size-1 box), one region that meets no group, a flat unit and points in no region, perturbed at size eps:
   (i)   at eps = 0 the assembly B^0 has loss exactly sum L^0 (TM1 identity),
   (ii)  sum L^0 <= V_rob(fixed boxes) <= sup_U D(B^0) <= sum L^0 + E, at several eps,
   (iii) E / eps stays bounded (district leaks: linear; the others: their errors are squared, so E is linear in the
         perturbation size too),
   (iv)  V_b on N's own table (fixed boxes) can be below sum L^0 by an amount that does NOT go to 0 with eps
         (Proposition TE1 on random instances).
"""
import itertools
import json
import os

import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(20261028)
OUT = {}


def lse(Y):
    m = Y.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(Y - m).sum(axis=-1, keepdims=True)))[..., 0]


def kl_rows(Q, Y):
    """KL(q_x || softmax(y_x)) per row."""
    return (Q * np.log(Q)).sum(-1) - (Q * Y).sum(-1) + lse(Y)


def softmax(Y):
    return np.exp(Y - lse(Y)[..., None])


# ---------------------------------------------------------------- A: Lemma TE0
worst = 0.0
for _ in range(20000):
    n = rng.integers(2, 7)
    q = rng.dirichlet(np.ones(n) * rng.uniform(0.2, 3))
    q = np.clip(q, 1e-12, None); q /= q.sum()
    y = rng.normal(0, 3, n); d = rng.normal(0, rng.uniform(0.01, 10), n)
    lhs = abs(kl_rows(q[None], (y + d)[None])[0] - kl_rows(q[None], y[None])[0])
    worst = max(worst, lhs / (d.max() - d.min()))
OUT["A_TE0_max_ratio"] = worst
print(f"A  TE0: max |dKL| / osc(d) over 20000 draws = {worst:.6f}  ->", "HELD" if worst <= 1 + 1e-12 else "FAILED")


# ---------------------------------------------------------------- two-logit helpers (TM4 instances)
def sig(g):
    return 1 / (1 + np.exp(-g))


def kl2(q1, g):
    s1 = sig(g)
    return q1 * np.log(q1 / s1) + (1 - q1) * np.log((1 - q1) / (1 - s1))


def fit2(q1, F, w):
    th = np.zeros(F.shape[1])
    for _ in range(400):
        s1 = sig(F @ th)
        g = F.T @ (w * (s1 - q1))
        H = F.T @ (F * (w * s1 * (1 - s1))[:, None]) + 1e-15 * np.eye(len(th))
        step = np.linalg.lstsq(H, g, rcond=None)[0]
        th -= step
        if np.abs(step).max() < 1e-14:
            break
    return float(np.sum(w * kl2(q1, F @ th))), th


p3 = np.ones(3) / 3
one = np.ones(3)

# ---------------------------------------------------------------- B: Proposition TE1 on TM4 part 2
s = np.array([1, 0, 0.]); t1 = np.array([0, 1, 0.]); j2 = np.array([0, 1, 1.]); g = np.array([0, 1, 0.])
gapN = 1.0 * s + 2.0 * t1 - 0.3 * j2              # N's gap; N does not read j1
q1 = sig(gapN)
V0, _ = fit2(q1, np.c_[s, s, j2, one], p3)         # delta = 0: j1 = s
Vplus, _ = fit2(q1, np.c_[s, s, j2, g, one], p3)   # N^+: one more kept unit with values g
rowsB = []
okB = True
for delta in (0.3, 0.1, 0.01, 0.001):
    j1 = s + delta * g
    Vd, th = fit2(q1, np.c_[s, j1, j2, one], p3)
    eta = 1e-9
    bound = (V0 - Vd - eta) / (delta * float(p3 @ g))
    sw = abs(th[1])                                   # swing of j1 in a two-logit (gap) build = |gap weight|
    rowsB.append(dict(delta=delta, V_b=Vd, V_plus=Vplus, V0=V0, swing_j1=sw, swing_bound=bound))
    okB &= abs(Vd - Vplus) < 1e-9 and sw >= bound - 1e-6
    print(f"B  TE1 delta={delta}: V_b={Vd:.2e} V_b(N+)={Vplus:.2e} V_b(N_0)={V0:.6f} swing(j1)={sw:.4g} >= {bound:.4g}")
print("B  TE1 parts 2-3:", "HELD" if okB else "FAILED")
OUT["B_TE1"] = rowsB


# ---------------------------------------------------------------- robust fit: inf over heads of sum_x p_x max_v loss
def robust_fit_gap(q1, Fv, w):
    """Fv[x] = list of feature rows (one per vertex of U at x). Minimise sum_x w_x max_v kl2(q1_x, Fv[x][v] . th)
    via the epigraph form (convex)."""
    nf = Fv[0].shape[1]
    nx = len(Fv)

    def obj(z):
        return float(w @ z[nf:])

    cons = []
    for x in range(nx):
        for v in range(Fv[x].shape[0]):
            cons.append(dict(type="ineq", fun=(lambda z, x=x, v=v: z[nf + x] - kl2(q1[x], Fv[x][v] @ z[:nf]))))
    best = None
    for start in range(2):
        z0 = np.r_[rng.normal(0, 1, nf), np.ones(nx) * 2]
        r = minimize(obj, z0, constraints=cons, method="SLSQP", options=dict(maxiter=2000, ftol=1e-14))
        val = max(float(w @ np.array([max(kl2(q1[x], Fv[x][v] @ r.x[:nf]) for v in range(Fv[x].shape[0]))
                                      for x in range(nx)])), 0.0)
        best = val if best is None else min(best, val)
    return best


# ---------------------------------------------------------------- C: Corollary TE4.1 part 2
LT, _ = fit2(q1, np.c_[j2, one], p3 * np.array([0, 1, 1.]))
rowsC = []
okC = True
for delta in (0.3, 0.1, 0.01, 0.001):
    # features (s, j1, j2, 1); only h_j1(b) moves, between 0 (clean) and delta
    Fv = [np.array([[1, 1, 0, 1.]]), np.array([[0, 0, 1, 1.], [0, delta, 1, 1.]]), np.array([[0, 0, 1, 1.]])]
    Vr = robust_fit_gap(q1, Fv, p3)
    Vb, _ = fit2(q1, np.c_[s, s + delta * g, j2, one], p3)
    # TE3 part 3: the worst case over the interval is at an end (check on a grid for a fixed head)
    th = rng.normal(0, 3, 4)
    grid = [kl2(q1[1], np.array([0, v, 1, 1.]) @ th) for v in np.linspace(0, delta, 101)]
    vert = max(grid[0], grid[-1]) >= max(grid) - 1e-15
    rowsC.append(dict(delta=delta, V_rob=Vr, L_T=LT, V_b=Vb, vertex_max=bool(vert)))
    okC &= abs(Vr - LT) < 1e-7 and Vb < 1e-9 and vert
    print(f"C  TE4.1(2) delta={delta}: V_rob={Vr:.8f} L_T={LT:.8f} V_b={Vb:.1e} worst-at-vertex={vert}")
print("C  Corollary TE4.1 part 2:", "HELD" if okC else "FAILED")
OUT["C_TE41_part2"] = rowsC

# ---------------------------------------------------------------- D: Corollary TE4.1 part 1
j1d = np.array([1, 0, 0.])
gapN1 = 1.0 * s + 0.5 * j1d + 2.0 * t1 - 0.3 * j2
q1d = sig(gapN1)
LT1, _ = fit2(q1d, np.c_[j2, one], p3 * np.array([0, 1, 1.]))
rowsD = []
okD = True
for delta in (0.3, 0.1, 0.01):
    zs = np.array([1, -1, -1 + delta])
    zbar = (zs[1] + zs[2]) / 2                       # clean shelter value (p uniform)
    # V_b = 0 via the box relu((z_s + 1) / delta): exact fit with box, j1, j2, bias
    box = np.maximum((zs + 1) / delta, 0)
    Vb, _ = fit2(q1d, np.c_[box, j1d, j2, one], p3)
    # copy box relu(z_s): worst case over U (z_s(b), z_s(c) in their intervals); box = 0 there on all of U
    zb_int = np.linspace(min(zs[1], zbar), max(zs[1], zbar), 7)
    zc_int = np.linspace(min(zs[2], zbar), max(zs[2], zbar), 7)
    Fv = [np.array([[1, 1, 0, 1.]]),
          np.array([[max(zb, 0), 0, 1, 1.] for zb in zb_int]),
          np.array([[max(zc, 0), 0, 1, 1.] for zc in zc_int])]
    Vcopy = robust_fit_gap(q1d, Fv, p3)
    # search: boxes relu(m z + beta), heads fitted to the worst case over a grid of U
    lowest = np.inf
    for _ in range(20):
        m, beta = rng.normal(0, 1 / delta), rng.normal(0, 1 / delta)
        Fv = [np.array([[max(m * zs[0] + beta, 0), 1, 0, 1.]]),
              np.array([[max(m * zb + beta, 0), 0, 1, 1.] for zb in zb_int]),
              np.array([[max(m * zc + beta, 0), 0, 1, 1.] for zc in zc_int])]
        lowest = min(lowest, robust_fit_gap(q1d, Fv, p3))
    rowsD.append(dict(delta=delta, V_b=Vb, V_rob_copy=Vcopy, L_T=LT1, lowest_found=lowest))
    okD &= Vb < 1e-9 and abs(Vcopy - LT1) < 1e-7 and lowest >= LT1 - 1e-7
    print(f"D  TE4.1(1) delta={delta}: V_b={Vb:.1e} copy-box worst={Vcopy:.8f} L_T={LT1:.8f} lowest found={lowest:.8f}")
print("D  Corollary TE4.1 part 1:", "HELD" if okD else "FAILED")
OUT["D_TE41_part1"] = rowsD

# ---------------------------------------------------------------- F: Proposition TE5 (the referee's instance)
def bkl(a, b):
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


rowsF = []
okF = True
target = 0.4 * bkl(sig(0.5), 0.5)
for W in (1e2, 1e4, 1e6):
    dl = W ** -0.5
    # units at x0, x1, x1', x2, x3
    j = np.array([0, 1, 1 + 1 / W, dl, 0]); k1 = np.array([0, 1, 1, 0, 0.]); k2 = np.array([0, 0, 0, 1, 0.])
    k3 = np.array([0, 0, 0, 0, 1.]); zs = np.array([-1, -1, -1, -1, 1.]); box = np.maximum(zs, 0)
    gap = W * j + (-W - 0.5) * k1 + (-W * dl) * k2
    q = sig(gap); p5 = np.ones(5) / 5
    # B^0: j, k1, k2 with N's weights (their regions meet no group); the copy box of s fits x3
    g_clean = W * np.r_[0, 1, 1 + 1 / W, 0, 0] + (-W - 0.5) * k1 + (-W * dl) * k2
    # stable form: KL(Bern(q) || Bern(sig(g))) = q (log q + log(1 + e^-g)) + (1 - q) (log(1 - q) + log(1 + e^g))
    D_B0_clean = float(p5 @ (q * (np.log(q) + np.logaddexp(0, -g_clean)) + (1 - q) * (np.log(1 - q) + np.logaddexp(0, g_clean))))
    # robust optimum: free head on (j, k1, k2, k3, box, 1); only h_j(x2) moves, between 0 and delta
    feats = np.c_[j, k1, k2, k3, box, np.ones(5)]
    Fv = []
    for x in range(5):
        if x == 3:
            f0 = feats[x].copy(); f0[0] = 0.0
            Fv.append(np.array([f0, feats[x]]))
        else:
            Fv.append(feats[x][None])
    Vr = robust_fit_gap(q, Fv, p5)
    # the lower bound of the proof, on a grid of w_j
    phi = lambda t: bkl(0.5, sig(t))
    cs = np.linspace(-2, 2, 801)
    psi = lambda t: min(bkl(sig(-0.5), sig(c)) + bkl(sig(0.5), sig(c + t)) for c in cs)
    wgrid = np.r_[-np.logspace(-3, np.log10(W ** 1.2), 200)[::-1], 0, np.logspace(-3, np.log10(W ** 1.2), 200)]
    lb = min(0.2 * (phi(abs(w) * dl / 2) + psi(w / W)) for w in wgrid)
    eps_dist = 0.2 * dl / float(p5 @ j)
    eps_quiet = 0.2 * W * dl
    rowsF.append(dict(W=W, gaps=gap.tolist(), V_rob_found=Vr, lower_bound_grid=lb, limit=target,
                      D_B0_clean=D_B0_clean, eps_dist=eps_dist, eps_quiet=eps_quiet))
    okF &= lb - 1e-6 <= Vr <= target + 1e-7  # the grid minimum can sit a little above the true minimum and np.allclose(gap, [0, -0.5, 0.5, 0, 0], atol=1e-6)
    print(f"F  TE5 W={W:.0e}: gaps {np.round(gap, 6).tolist()} eps_dist={eps_dist:.2e} eps_quiet={eps_quiet:.3g} | "
          f"lower {lb:.6f} <= V_rob found {Vr:.6f} <= limit {target:.6f} | D(B0) on the clean table {D_B0_clean:.3f}")
okF &= rowsF[-1]["lower_bound_grid"] > 0.9 * target and rowsF[-1]["D_B0_clean"] > rowsF[0]["D_B0_clean"] * 50
print("F  Proposition TE5:", "HELD" if okF else "FAILED")
OUT["F_TE5"] = rowsF

allok = worst <= 1 + 1e-12 and okB and okC and okD and okF
print("A-F ALL:", "HELD" if allok else "SOME FAILED")
json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "te_robust.json"), "w"), indent=1,
          default=float)
