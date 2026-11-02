# serves: TP2
"""Theorems TP1, TP2, Corollary TP1.1 and Proposition TP3 (COARSE-THEORY.md B.P): summaries passed up the tree.
Exact part (fractions, sympy):
  A. Proposition TH3's instance: summaries of w_1, w_2 (DP2), eps_lin = 0, Lin(G_1) = 3/2, Lin(G_2) = 2, the
     compressed summary Sigma'_1 = 4 ehat_a ehat_a^T gives 3, dropped trace 3, so 3 <= 2 + 3/2 (TP2 parts 3, 4).
  B. Proposition TH2 part 2's instance: Lin = 1/20, eps_lin = 1/10 at G_1 and G_2; Lin = 0, eps_lin = 0 at G_3
     (TP2 part 5).
Numeric part (random instances that meet TH1's premises; checked, not assumed):
  C. TP1 part 1: no ReLU box found by search beats Lin_S - eps_lin(S) / 2 (the lower bound).
  D. TP1 part 2: the always-on box built from the summaries reaches Lin_S exactly.
  E. TP2 part 1: Sigma of a merged group = sum of its nodes' Sigma; a level from summaries = a level from data.
  F. TP2 part 3: a build from compressed summaries has loss (1/2)(E - tr Q'^T Sigma Q') <= Lin + tau / 2.
  G. TP3: time of all summaries against one forward pass; time of a level from summaries.
"""
import json
import os
import time
from fractions import Fraction as Fr

import numpy as np
import sympy as sp
from scipy.optimize import minimize

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tp_summary.json")
fails = []
report = {}


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + ("  " + str(detail) if detail != "" else ""))
    if not ok:
        fails.append(name)


# ---------------------------------------------------------------- exact summaries (sympy)
def summary_exact(Y, Z, H, w):
    """DP1/DP2 on one area, exactly. Y: list of n_y-vectors (already P_perp), Z: top pre-activations, H: judgment
    values, w: weights. Returns E, Sigma, eps, B."""
    n = len(Y)
    W = sp.diag(*[sp.Rational(x) for x in w])
    F = sp.Matrix([[1] + list(h) for h in H])                   # basis of F_w (constant, h_J)
    G = F.T * W * F
    Pi = sp.eye(n) - F * G.pinv() * F.T * W                     # projection onto the complement of F_w in L2(w)
    Yt = Pi * sp.Matrix(Y)
    Zt = Pi * sp.Matrix(Z)
    Czz = Zt.T * W * Zt
    Cyz = Yt.T * W * Zt
    B = Cyz * Czz.pinv()
    R = Yt - Zt * B.T
    E = sum(w[i] * (Yt[i, :] * Yt[i, :].T)[0] for i in range(n))
    eps = sum(w[i] * (R[i, :] * R[i, :].T)[0] for i in range(n))
    Sigma = B * Czz * B.T
    return sp.nsimplify(E), sp.simplify(Sigma), sp.nsimplify(eps), B


def pperp(v):
    m = sum(v) / len(v)
    return [a - m for a in v]


def topk_exact(M, k):
    ev = sorted([sp.nsimplify(e) for e, mult in M.eigenvals().items() for _ in range(mult)], reverse=True)
    return sum(ev[:k])


# A. Proposition TH3's instance
ea, eb = [1, -1, 0], [1, 1, -2]
pts1 = [(3 + sa, 3 + sb) for sa, sb in ((2, 1), (2, -1), (-2, 1), (-2, -1))]
Y1 = [pperp([ea[i] * xa + eb[i] * xb + eb[i] * 0 for i in range(3)]) for xa, xb in pts1]      # u_c off on Omega_1
Y2 = [pperp([eb[i] * (3 + sc) for i in range(3)]) for sc in (1, -1)]                           # u_a, u_b off on Omega_2
E1, S1, e1, B1 = summary_exact(Y1, [list(p) for p in pts1], [[1]] * 4, [Fr(1, 8)] * 4)
E2, S2, e2, B2 = summary_exact(Y2, [[3 + sc] for sc in (1, -1)], [[1]] * 2, [Fr(1, 4)] * 2)
check("A eps_lin(w_1) = eps_lin(w_2) = 0", e1 == 0 and e2 == 0, (e1, e2))
check("A E_1 = 7, E_2 = 3", (E1, E2) == (7, 3), (E1, E2))
Ehat_a = sp.Matrix(ea) / sp.sqrt(2)
Ehat_b = sp.Matrix(eb) / sp.sqrt(6)
check("A Sigma_1 = 4 ehat_a ehat_a^T + 3 ehat_b ehat_b^T", sp.simplify(S1 - 4 * Ehat_a * Ehat_a.T - 3 * Ehat_b * Ehat_b.T) == sp.zeros(3))
check("A Sigma_2 = 3 ehat_b ehat_b^T", sp.simplify(S2 - 3 * Ehat_b * Ehat_b.T) == sp.zeros(3))
lin_g1 = sp.Rational(1, 2) * (E1 - topk_exact(S1, 1)) + sp.Rational(1, 2) * (E2 - topk_exact(S2, 1))
lin_g2 = sp.Rational(1, 2) * (E1 + E2 - topk_exact(S1 + S2, 1))
check("A Lin(G_1) = 3/2 (= V2_b(G_1) of TH3)", lin_g1 == sp.Rational(3, 2), lin_g1)
check("A Lin(G_2) = 2 (= V2_b(G_2) of TH3)", lin_g2 == 2, lin_g2)
S1c = 4 * Ehat_a * Ehat_a.T
tau = sp.nsimplify(sp.simplify((S1 - S1c).trace()))
comp = sp.Rational(1, 2) * (E1 + E2 - topk_exact(sp.simplify(S1c + S2), 1))
check("A compressed summary (child 1's least unit) gives 3, tau = 3", comp == 3 and tau == 3, (comp, tau))
check("A TP2 part 3 bound: 3 <= 2 + 3/2", comp <= lin_g2 + tau / 2)
check("A Sigma_1 - Sigma'_1 is positive semidefinite", all(sp.nsimplify(e) >= 0 for e in (S1 - S1c).eigenvals()))
report["A"] = {"Lin_G1": str(lin_g1), "Lin_G2": str(lin_g2), "compressed": str(comp), "tau": str(tau)}

# B. Proposition TH2 part 2's instance: P = (1,0),(2,1),(3,0),(4,1), gap Delta = f1 - 2 f2 + 2 f3, y = (Delta, 0)
P = [(1, 0), (2, 1), (3, 0), (4, 1)]
wB = [Fr(1, 4)] * 4
gap = []
for x1, x2 in P:
    g = max(x1, 0)
    f = [max(g - i, 0) for i in (1, 2, 3)]
    gap.append(f[0] - 2 * f[1] + 2 * f[2])
check("B gap on P is 0, 1, 0, 1", gap == [0, 1, 0, 1], gap)
YB = [pperp([d, 0]) for d in gap]
EB1, SB1, eB1, _ = summary_exact(YB, [[x1 - i for i in (1, 2, 3)] for x1, _ in P], [[]] * 4, wB)   # G_1: top f (z = g - i)
EB2, SB2, eB2, _ = summary_exact(YB, [[x1] for x1, _ in P], [[]] * 4, wB)                         # G_2: top g (z = a_1)
EB3, SB3, eB3, _ = summary_exact(YB, [[x1, x2] for x1, x2 in P], [[]] * 4, wB)                     # G_3: top a_1, a_2
linB = [sp.Rational(1, 2) * (E - topk_exact(S, 1)) for E, S in ((EB1, SB1), (EB2, SB2), (EB3, SB3))]
check("B E = 1/8 at every level", EB1 == EB2 == EB3 == sp.Rational(1, 8), (EB1, EB2, EB3))
check("B Lin(G_1) = Lin(G_2) = 1/20, eps_lin = 1/10", linB[0] == linB[1] == sp.Rational(1, 20) and eB1 == eB2 == sp.Rational(1, 10),
      (linB, eB1, eB2))
check("B Lin(G_3) = 0, eps_lin = 0", linB[2] == 0 and eB3 == 0, (linB[2], eB3))
report["B"] = {"Lin": [str(v) for v in linB], "eps": [str(eB1), str(eB2), str(eB3)]}


# ---------------------------------------------------------------- random instances (numpy)
def make_instance(rng, n_areas, per_area, d_feat, top, n_y, deep, switch):
    """Input = (one-hot gate j, one feature block per area). Outside its area a block is -1. Layer 1: judgment units
    J_i = relu(j_i), top units of area i: relu(W_i (f_i + 1) - 1) (pre-activation -1, off, outside the area). With
    deep, layer 2 of area i: relu(A_i h1_i + c_i), c_i <= 0. Logits read the last layer of each area and J."""
    X, area = [], []
    for i in range(n_areas):
        for _ in range(per_area):
            j = np.zeros(n_areas); j[i] = 1
            f = -np.ones((n_areas, d_feat))
            f[i] = rng.uniform(0.5, 2.0, d_feat)
            X.append((j, f)); area.append(i)
    area = np.array(area)
    w = rng.uniform(0.5, 1.5, len(X)); w /= w.sum()
    Ws = [(rng.uniform(0.2, 1.0, (top, d_feat)) if not switch else rng.normal(0, 1, (top, d_feat))) for _ in range(n_areas)]
    As = [(rng.uniform(0.0, 1.0, (top, top)) if not switch else rng.normal(0, 1, (top, top))) for _ in range(n_areas)]
    cs = [(np.zeros(top) if not switch else -rng.uniform(0, 1, top)) for _ in range(n_areas)]
    Wo = [rng.normal(0, 1, (n_y, top)) for _ in range(n_areas)]
    wJ = rng.normal(0, 1, (n_y, n_areas))
    Z1 = np.zeros((len(X), n_areas, top)); HJ = np.zeros((len(X), n_areas)); Y = np.zeros((len(X), n_y))
    for p, (j, f) in enumerate(X):
        HJ[p] = np.maximum(j, 0)
        y = wJ @ HJ[p]
        for i in range(n_areas):
            z = Ws[i] @ (f[i] + 1) - 1
            Z1[p, i] = z
            h = np.maximum(z, 0)
            if deep:
                h = np.maximum(As[i] @ h + cs[i], 0)
            y = y + Wo[i] @ h
        Y[p] = y - y.mean()                                    # P_perp y_N
    return dict(Z=Z1, HJ=HJ, Y=Y, w=w, area=area, n_areas=n_areas, top=top, n_y=n_y)


def claims_ok(inst):
    """District, shelter and judgment claims at error 0 (checked on the data)."""
    Z, HJ, area = inst["Z"], inst["HJ"], inst["area"]
    ok = True
    for i in range(inst["n_areas"]):
        out = area != i
        ok &= bool(np.all(Z[out, i] <= 0))                     # district: off outside the area
        ok &= bool(np.allclose(Z[out, i], Z[out, i][0]))       # shelter: constant outside the area
        ok &= bool(np.allclose(HJ[area == i, i], 1)) and bool(np.allclose(HJ[out, i], 0))  # judgment J_i = 1 on its area
    return ok


def summary(inst, i):
    """DP1/DP2 for the node of area i (top = its layer-1 units, J = {J_i}, constant 1 on the area)."""
    m = inst["area"] == i
    w = inst["w"][m]
    Y, Z = inst["Y"][m], inst["Z"][m, i]
    Yt = Y - w @ Y / w.sum()                                   # F_w = span{1} since h_J = 1 on the area
    Zt = Z - w @ Z / w.sum()
    Czz = Zt.T @ (w[:, None] * Zt)
    Cyz = Yt.T @ (w[:, None] * Zt)
    B = Cyz @ np.linalg.pinv(Czz)
    R = Yt - Zt @ B.T
    E = float(np.sum(w[:, None] * Yt ** 2))
    eps = float(np.sum(w[:, None] * R ** 2))
    return dict(E=E, Sigma=B @ Czz @ B.T, eps=eps, B=B)


def topk(M, k):
    ev = np.sort(np.linalg.eigvalsh(M))[::-1]
    return float(np.sum(ev[:k]))


def group_loss(inst, nodes, C):
    """Terms of L2_S for box values C (points x k) with the least V (shared), U_w, t_w (per area)."""
    Ys, Cs, ws = [], [], []
    for i in nodes:
        m = inst["area"] == i
        w = inst["w"][m]
        Ys.append(inst["Y"][m] - w @ inst["Y"][m] / w.sum())
        Cs.append(C[m] - w @ C[m] / w.sum())
        ws.append(w)
    Y, C2, w = np.vstack(Ys), np.vstack(Cs), np.concatenate(ws)
    Ccc = C2.T @ (w[:, None] * C2)
    Cyc = Y.T @ (w[:, None] * C2)
    V = Cyc @ np.linalg.pinv(Ccc)
    return 0.5 * float(np.sum(w[:, None] * (Y - C2 @ V.T) ** 2))


def top_values(inst, nodes):
    return np.concatenate([inst["Z"][:, i] for i in nodes], axis=1)   # z_T(S) at every point of P


def box_values(zT, m, beta):
    return np.maximum(zT @ m.T + beta, 0)


def linear_box(inst, nodes, S, k, Q=None):
    Sig = sum(S[i]["Sigma"] for i in nodes)
    if Q is None:
        ev, U = np.linalg.eigh(Sig)
        Q = U[:, np.argsort(ev)[::-1][:k]]
    m = np.concatenate([S[i]["B"].T @ Q for i in nodes], axis=0).T     # k x |T(S)|
    zT = top_values(inst, nodes)
    beta = -np.min(zT @ m.T, axis=0) + 1.0                              # on at every point of P
    return m, beta


def search_box(inst, nodes, k, rng, starts):
    zT = top_values(inst, nodes)
    nT = zT.shape[1]
    sel = np.isin(inst["area"], nodes)

    def f(theta):
        m = theta[: k * nT].reshape(k, nT); beta = theta[k * nT:]
        return group_loss(inst, nodes, box_values(zT, m, beta) * sel[:, None] + box_values(zT, m, beta) * ~sel[:, None])

    best = np.inf
    for s in range(starts):
        th0 = np.concatenate([rng.normal(0, 1, k * nT), rng.normal(0, 1, k)])
        r = minimize(f, th0, method="L-BFGS-B", options={"maxiter": 300})
        best = min(best, r.fun)
    return best


rng = np.random.default_rng(0)
rows = []
cases = [  # (n_areas, per_area, d_feat, top, n_y, deep, switch, k, group nodes)
    (2, 8, 4, 2, 3, False, False, 1, [0, 1]),
    (3, 8, 4, 3, 4, False, False, 2, [0, 1, 2]),
    (3, 8, 4, 3, 4, True, False, 2, [0, 1, 2]),
    (2, 10, 4, 3, 3, False, True, 1, [0, 1]),
    (3, 8, 4, 3, 4, False, True, 2, [0, 1, 2]),
    (3, 8, 4, 3, 4, True, True, 2, [0, 1, 2]),
    (2, 10, 3, 2, 5, True, True, 2, [0, 1]),
]
t0 = time.time()
for trial in range(3):
    for (na, pa, df, top, ny, deep, sw, k, nodes) in cases:
        inst = make_instance(rng, na, pa, df, top, ny, deep, sw)
        if not claims_ok(inst):
            check(f"C claims hold (case {na},{deep},{sw})", False)
            continue
        S = {i: summary(inst, i) for i in range(na)}
        E = sum(S[i]["E"] for i in nodes)
        Sig = sum(S[i]["Sigma"] for i in nodes)
        lin = 0.5 * (E - topk(Sig, k))
        eps = sum(S[i]["eps"] for i in nodes)
        # D: the always-on box reaches Lin
        m, beta = linear_box(inst, nodes, S, k)
        zT = top_values(inst, nodes)
        lb = group_loss(inst, nodes, box_values(zT, m, beta))
        on_everywhere = bool(np.all(zT @ m.T + beta > 0))
        # D': the paper's bias beta_t = 1 + sum rho_w (rho_w = max over P of |B_w z_T_w|) keeps every unit on
        rho = sum(float(np.max(np.linalg.norm(inst["Z"][:, i] @ S[i]["B"].T, axis=1))) for i in nodes)
        check("D' paper bias 1 + sum rho_w: units on at every point", bool(np.all(zT @ m.T + 1 + rho >= 1 - 1e-9)))
        # TP1.1: boxes of always-on units (any directions, any m) never go below Lin
        worst = np.inf
        for _ in range(200):
            mr = rng.normal(0, 1, m.shape)
            br = -np.min(zT @ mr.T, axis=0) + 1.0
            worst = min(worst, group_loss(inst, nodes, box_values(zT, mr, br)))
        check("TP1.1 always-on boxes never below Lin", worst >= lin - 1e-8, (round(worst, 6), round(lin, 6)))
        # C: search over ReLU boxes
        best = search_box(inst, nodes, k, rng, starts=6)
        best = min(best, lb)
        row = dict(areas=na, deep=deep, switch=sw, k=k, Lin=lin, eps=eps, linear_box=lb, best_search=best)
        rows.append(row)
        check(f"D linear box on everywhere, loss = Lin  (deep={deep}, switch={sw}, k={k})",
              on_everywhere and abs(lb - lin) <= 1e-8 * max(1, lin), (round(lb, 10), round(lin, 10)))
        check(f"C search never below Lin - eps/2  (deep={deep}, switch={sw})", best >= lin - eps / 2 - 1e-8,
              (round(best, 6), round(lin - eps / 2, 6), "eps", round(eps, 6)))
        if not sw:
            check(f"C eps_lin = 0 without switching (deep={deep})", eps <= 1e-10, eps)
        # E: additivity, and a level from summaries = a level from data
        if len(nodes) >= 2:
            a, b = nodes[:1], nodes[1:]
            Sa = sum(S[i]["Sigma"] for i in a); Sb = sum(S[i]["Sigma"] for i in b)
            check("E Sigma of the merge = sum of the parts", np.allclose(Sa + Sb, Sig))
        # F: compressed summaries
        Sc = {}
        tau = 0.0
        for i in nodes:
            ev, U = np.linalg.eigh(S[i]["Sigma"])
            keep = ev >= np.quantile(ev, 0.6)                        # drop the smaller eigen-parts
            Sc[i] = (U[:, keep] * ev[keep]) @ U[:, keep].T
            tau += float(np.sum(ev[~keep].clip(min=0)))
        evc, Uc = np.linalg.eigh(sum(Sc[i] for i in nodes))
        Qc = Uc[:, np.argsort(evc)[::-1][:k]]
        mc, bc = linear_box(inst, nodes, S, k, Q=Qc)
        lc = group_loss(inst, nodes, box_values(zT, mc, bc))
        formula = 0.5 * (E - float(np.trace(Qc.T @ Sig @ Qc)))
        check("F compressed build: loss <= (1/2)(E - tr Q'^T Sigma Q') <= Lin + tau/2",
              lc <= formula + 1e-8 and formula <= lin + tau / 2 + 1e-8, (round(lc, 6), round(formula, 6), round(lin + tau / 2, 6)))
report["random"] = rows
report["random_seconds"] = time.time() - t0

# G. cost (TP3): summaries of every node against one forward pass, at a larger size
rngG = np.random.default_rng(1)
na, per, df, top, ny = 8, 400, 64, 64, 1000
n = na * per
X = rngG.normal(0, 1, (n, df * na))
W1 = rngG.normal(0, 1, (top * na, df * na)) / np.sqrt(df * na)
W2 = rngG.normal(0, 1, (top * na, top * na)) / np.sqrt(top * na)
Wo = rngG.normal(0, 1, (ny, top * na))
t = time.time()
Z1 = X @ W1.T; H2 = np.maximum(np.maximum(Z1, 0) @ W2.T, 0); Yb = H2 @ Wo.T; Yb -= Yb.mean(1, keepdims=True)
t_fwd = time.time() - t
area = np.repeat(np.arange(na), per)
wG = np.full(n, 1 / n)
t = time.time()
facs = []
for i in range(na):
    msk = area == i
    Yt = Yb[msk] - Yb[msk].mean(0); Zt = Z1[msk][:, i * top:(i + 1) * top]; Zt = Zt - Zt.mean(0)
    Czz = Zt.T @ (wG[msk][:, None] * Zt); Cyz = Yt.T @ (wG[msk][:, None] * Zt)
    ev, U = np.linalg.eigh(Czz)
    Cinvh = (U / np.sqrt(np.clip(ev, 1e-12, None))) @ U.T
    facs.append(Cyz @ Cinvh)                                  # F_w = B_w C_zz^(1/2) = C_yz C_zz^(-1/2)
t_sum = time.time() - t
t = time.time()
Fm = np.concatenate(facs, axis=1)
Uf, sv, _ = np.linalg.svd(Fm, full_matrices=False)
lin_top = float(np.sum(sv[:16] ** 2))
t_level = time.time() - t
report["cost"] = dict(points=n, units=2 * top * na, logits=ny, forward_s=t_fwd, summaries_s=t_sum, level_s=t_level,
                      factor_shape=list(Fm.shape))
print(f"G cost: forward pass {t_fwd:.3f}s, all {na} summaries {t_sum:.3f}s, one level (merge of {na}, k=16) {t_level:.3f}s")
check("G a level from summaries does not touch N (factor has n_y rows, sum of top sizes columns)", Fm.shape == (ny, na * top))

report["fails"] = fails
json.dump(report, open(OUT, "w"), indent=1, default=float)
print(f"\n{len(fails)} failures")
