# serves: TL1
# serves: TL2
"""Theorems TL1, TL2 and Corollary TL2.1 (COARSE-THEORY.md B.L): the broken cases read as networks.
Part A (TL1, one hidden layer, D2, linear-read claims at eps 0):
  A1 the instance of Proposition TH3: condition (K) fails at the merge (exact, sympy).
  A2 the same network with k_1 = 2: (K) holds by part 2(a), and the (r) step of the proof reaches Lin' = 2.
  A3 random networks with children that write orthogonal output directions and k' <= min k_c: (K) holds by part 2(b),
     and the (r) step of the proof reaches Lin' (the least, TP1 part 3). Random networks without orthogonality: (K)
     is tested and the (r) value is reported (no claim is made there).
  A4 the graded form: orthogonal children with a nonlinear part (eps_lin > 0) and sizes up to n_y + 1 (zero padding):
     the (r) step reaches Lin' exactly.
Part B (TL2):
  B1 the instance of TH2 part 2 (exact fractions): every finer box unit equals the lifted unit of (**) at every point.
  B2 random two-hidden-layer networks with a node u whose units are on in its area, a flat judgment unit, shelter
     constants: f = g + (c_0 - c_1)(1 - sum a h_K) at every point of P, for random finer box units f.
A failure is a case where the theorem's conclusion does not hold within 1e-9."""
import json
import os
from fractions import Fraction as Fr

import numpy as np
import sympy as sp

rng = np.random.default_rng(29)
fails = []
out = {}


def Pperp(n):
    return np.eye(n) - np.ones((n, n)) / n


def perp_basis(n):
    # orthonormal basis of the image of P_perp
    U, _, _ = np.linalg.svd(Pperp(n))
    return U[:, : n - 1]


# ---------- Part A: one hidden layer, areas, judgment units constant on their areas ----------
def summaries(nodes, p, Y, Z):
    """nodes: list of (pts, units). Returns per node B_w, Sigma_w, E_w, eps_lin (F_w = constants on the area)."""
    res = []
    for pts, units in nodes:
        w = p[pts] / p[pts].sum()
        Yt = Y[pts] - w @ Y[pts]
        Zt = Z[np.ix_(pts, units)] - w @ Z[np.ix_(pts, units)]
        pw = p[pts]
        Czz = (Zt * pw[:, None]).T @ Zt
        Cyz = (Yt * pw[:, None]).T @ Zt
        B = Cyz @ np.linalg.pinv(Czz)
        r = Yt - Zt @ B.T
        E = float(np.sum(pw[:, None] * Yt ** 2))
        res.append(dict(B=B, S=B @ Czz @ B.T, E=E, eps=float(np.sum(pw[:, None] * r ** 2))))
    return res


def top_vecs(S, k, ny):
    U = perp_basis(ny)
    ev, V = np.linalg.eigh(U.T @ S @ U)
    order = np.argsort(-ev)
    Q = U @ V[:, order]
    if k > ny - 1:                                            # zero padding after n_y - 1 directions (TL1)
        Q = np.concatenate([Q, np.zeros((ny, k - ny + 1))], axis=1)
    return Q[:, :k], ev[order]


def group_terms(nodes, idx, p, Y, Z, C, V):
    """D2 terms of a group on its areas, with box values C (n x k), logit weights V (ny x k), free constants per area."""
    tot = 0.0
    ny = Y.shape[1]
    for i in idx:
        pts, _ = nodes[i]
        R = (Y[pts] - C[pts] @ V.T) @ Pperp(ny)
        w = p[pts] / p[pts].sum()
        R = R - w @ R
        tot += 0.5 * float(np.sum(p[pts][:, None] * R ** 2))
    return tot


def tp1_box(nodes, idx, summ, Z, k, ny):
    S = sum(summ[i]["S"] for i in idx)
    Q, ev = top_vecs(S, k, ny)
    M = np.zeros((Z.shape[1], k))
    for i in idx:
        M[nodes[i][1], :] = summ[i]["B"].T @ Q
    pre = Z @ M
    beta = 1 + np.abs(pre).max(axis=0)
    return dict(Q=Q, M=M, beta=beta, C=pre + beta, S=S, ev=ev)


def r_step(nodes, children, summ, Z, Qp):
    """The (r) step of TL1's proof: from children boxes (TP1 type) to the merged box with directions Qp.
    Returns the box values and the (K) residual (largest |Sigma_c (q' - Q_c alpha)|)."""
    kres = 0.0
    Mpp = np.zeros((Z.shape[1], Qp.shape[1]))
    for ch in children:
        Sc, Qc = ch["box"]["S"], ch["box"]["Q"]
        G = Qc.T @ Sc @ Qc
        alpha = np.linalg.pinv(G) @ Qc.T @ Sc @ Qp            # least |Sigma_c^(1/2) (q' - Q_c alpha)|
        kres = max(kres, float(np.abs(Sc @ (Qp - Qc @ alpha)).max()))
        Mpp += ch["box"]["M"] @ alpha                         # m'' = sum alpha m_(c,s), read through children's units
    pre = Z @ Mpp
    return pre + 1 + np.abs(pre).max(axis=0), kres


def lin_value(summ, idx, k, ny):
    S = sum(summ[i]["S"] for i in idx)
    E = sum(summ[i]["E"] for i in idx)
    _, ev = top_vecs(S, k, ny)
    return 0.5 * (E - ev[:k].clip(min=0).sum())


def make_net(areas, ny, write, bend=None):
    """areas: list of (n_points, n_units). write[a]: ny x n_units logit weights of area a's units.
    Units of area a: relu(feature) with feature = 3 + noise on area a, -1 elsewhere (district and shelter at eps 0)."""
    n = sum(a[0] for a in areas)
    nu = sum(a[1] for a in areas)
    Z = -np.ones((n, nu))
    p = rng.uniform(0.5, 1.5, n)
    p /= p.sum()
    nodes, r0, c0 = [], 0, 0
    for a, (npt, nun) in enumerate(areas):
        pts, units = np.arange(r0, r0 + npt), np.arange(c0, c0 + nun)
        Z[np.ix_(pts, units)] = 3 + rng.normal(0, 1, (npt, nun)).clip(-2.5, 2.5)
        nodes.append((pts, units))
        r0, c0 = r0 + npt, c0 + nun
    H = np.maximum(Z, 0)
    W = np.concatenate(write, axis=1)
    Y = H @ W.T
    if bend is not None:                                      # a nonlinear part: eps_lin > 0 (graded form of TL1)
        Y = Y + (H ** 2) @ np.concatenate(bend, axis=1).T
    return nodes, p, Y @ Pperp(ny), Z


def check_merge(nodes, p, Y, Z, groups, kc, kp, ny, label, expect_K, graded=False):
    summ = summaries(nodes, p, Y, Z)
    if not graded and max(s["eps"] for s in summ) > 1e-9:
        fails.append(f"{label}: eps_lin not 0")
    children = []
    for idx, k in zip(groups, kc):
        children.append(dict(idx=idx, box=tp1_box(nodes, idx, summ, Z, k, ny)))
    allidx = [i for g in groups for i in g]
    Sp = sum(summ[i]["S"] for i in allidx)
    Qp, _ = top_vecs(Sp, kp, ny)
    C, kres = r_step(nodes, children, summ, Z, Qp)
    val = group_terms(nodes, allidx, p, Y, Z, C, Qp)
    lin = lin_value(summ, allidx, kp, ny)
    K = kres < 1e-9
    if expect_K and not K:
        fails.append(f"{label}: (K) expected by part 2 but residual {kres:.2e}")
    if K and abs(val - lin) > 1e-9:
        fails.append(f"{label}: (K) holds but (r) value {val} != Lin' {lin}")
    return dict(K=K, kres=kres, r_value=val, lin=lin)


# A1, A2: Proposition TH3's network (J units are constant on their areas and add only constants)
def th3_net():
    ea, eb = np.array([1., -1, 0]), np.array([1., 1, -2])
    Z = np.array([[3 + sa, 3 + sb, -1] for sa, sb in ((2, 1), (2, -1), (-2, 1), (-2, -1))] +
                 [[-1, -1, 3 + sc] for sc in (1, -1)], float)
    p = np.array([1 / 8] * 4 + [1 / 4] * 2)
    Y = np.maximum(Z, 0) @ np.stack([ea, eb, eb], axis=1).T
    nodes = [(np.arange(0, 4), np.array([0, 1])), (np.arange(4, 6), np.array([2]))]
    return nodes, p, Y @ Pperp(3), Z


ea, eb = sp.Matrix([1, -1, 0]), sp.Matrix([1, 1, -2])
S1 = 2 * ea * ea.T + sp.Rational(1, 2) * eb * eb.T
S2 = sp.Rational(1, 2) * eb * eb.T
K1 = sp.Matrix.hstack(ea, sp.Matrix([1, 1, 1]))               # K_1 + ker Sigma_1
in_span = sp.Matrix.hstack(K1, eb).rank() == K1.rank()
out["A1"] = dict(eig_S1=str(S1.eigenvals()), eig_Sp=str((S1 + S2).eigenvals()), eb_in_K1_plus_ker=in_span)
if in_span:
    fails.append("A1: e_b lies in K_1 + ker Sigma_1, but TL1 part 4 says it does not")
nodes, p, Y, Z = th3_net()
out["A1_numeric"] = check_merge(nodes, p, Y, Z, [[0], [1]], [1, 1], 1, 3, "A1", expect_K=False)
if out["A1_numeric"]["K"]:
    fails.append("A1: numeric (K) test passed, expected to fail")
out["A2"] = check_merge(nodes, p, Y, Z, [[0], [1]], [2, 1], 1, 3, "A2", expect_K=True)

# A3: random networks
orth_rows, gen_rows = [], []
for trial in range(40):
    ny = int(rng.integers(4, 7))
    U = perp_basis(ny)
    nchild = int(rng.integers(2, 4))
    # split the perp space into orthogonal blocks, one per child
    dims = np.array_split(rng.permutation(ny - 1), nchild)
    areas, write, groups, kc = [], [], [], []
    node_id = 0
    for c in range(nchild):
        nn = int(rng.integers(1, 3))
        idx = []
        for _ in range(nn):
            nun = int(rng.integers(1, 4))
            areas.append((int(rng.integers(nun + 3, nun + 8)), nun))
            basis = U[:, dims[c]]
            write.append(basis @ rng.normal(0, 1, (len(dims[c]), nun)))
            idx.append(node_id)
            node_id += 1
        groups.append(idx)
        kc.append(int(rng.integers(1, 3)))
    kp = int(rng.integers(1, min(kc) + 1))
    nodes, p, Y, Z = make_net(areas, ny, write)
    orth_rows.append(check_merge(nodes, p, Y, Z, groups, kc, kp, ny, f"A3-orth-{trial}", expect_K=True))
    # same shapes, generic writes (no orthogonality)
    write = [rng.normal(0, 1, (ny, a[1])) for a in areas]
    nodes, p, Y, Z = make_net(areas, ny, write)
    gen_rows.append(check_merge(nodes, p, Y, Z, groups, kc, kp, ny, f"A3-gen-{trial}", expect_K=False))
out["A3_orth"] = dict(n=len(orth_rows), K_holds=sum(r["K"] for r in orth_rows),
                      max_gap=max(abs(r["r_value"] - r["lin"]) for r in orth_rows))
out["A3_generic"] = dict(n=len(gen_rows), K_holds=sum(r["K"] for r in gen_rows),
                         r_above_lin=sum(r["r_value"] > r["lin"] + 1e-9 for r in gen_rows),
                         r_equal_lin=sum(abs(r["r_value"] - r["lin"]) <= 1e-9 for r in gen_rows))


# A4: graded form (eps_lin > 0) and sizes at or above n_y (zero padding), orthogonal children
grad_rows = []
for trial in range(40):
    ny = int(rng.integers(3, 6))
    U = perp_basis(ny)
    nchild = int(rng.integers(2, 4))
    dims = np.array_split(rng.permutation(ny - 1), nchild)
    areas, write, bend, groups, kc = [], [], [], [], []
    node_id = 0
    for c in range(nchild):
        idx = []
        for _ in range(int(rng.integers(1, 3))):
            nun = int(rng.integers(1, 4))
            areas.append((int(rng.integers(nun + 3, nun + 8)), nun))
            write.append(U[:, dims[c]] @ rng.normal(0, 1, (len(dims[c]), nun)))
            bend.append(0.3 * U[:, dims[c]] @ rng.normal(0, 1, (len(dims[c]), nun)))
            idx.append(node_id)
            node_id += 1
        groups.append(idx)
        kc.append(int(rng.integers(1, ny + 2)))
    kp = int(rng.integers(1, min(kc) + 1))
    nodes, p, Y, Z = make_net(areas, ny, write, bend)
    row = check_merge(nodes, p, Y, Z, groups, kc, kp, ny, f"A4-{trial}", expect_K=True, graded=True)
    row["eps"] = sum(sm["eps"] for sm in summaries(nodes, p, Y, Z))
    grad_rows.append(row)
out["A4_graded"] = dict(n=len(grad_rows), K_holds=sum(r["K"] for r in grad_rows),
                        min_eps=min(r["eps"] for r in grad_rows),
                        max_gap_r_vs_Lin=max(abs(r["r_value"] - r["lin"]) for r in grad_rows))


# ---------- Part B: lifts ----------
def relu(t):
    return t if t > 0 else 0 * t


# B1: TH2 part 2, exact
P = [(Fr(1), Fr(0)), (Fr(2), Fr(1)), (Fr(3), Fr(0)), (Fr(4), Fr(1))]
b1_bad = 0
for _ in range(200):
    m = [Fr(int(v), 7) for v in rng.integers(-20, 21, 3)]
    beta = Fr(int(rng.integers(-30, 31)), 5)
    mu, nu = Fr(int(rng.integers(-20, 21)), 3), Fr(int(rng.integers(-30, 31)), 4)
    for x1, x2 in P:
        a1, a2 = relu(x1), relu(x2)
        g = relu(a1)
        # G_1 box unit reads (g - 1, g - 2, g - 3); lifted to G_2: reads z_g = a1
        f1 = relu(sum(mi * (g - i - 1) for i, mi in enumerate(m)) + beta)
        g1 = relu(sum(m) * a1 + (beta - sum(mi * (i + 1) for i, mi in enumerate(m))))
        # G_2 box unit reads z_g = a1; lifted to G_3: reads (z_a1, z_a2) = (x1, x2)
        f2 = relu(mu * a1 + nu)
        g2 = relu(mu * x1 + 0 * x2 + nu)
        b1_bad += (f1 != g1) + (f2 != g2)
out["B1"] = dict(trials=200, mismatches=int(b1_bad), pre_activations_nonneg=all(x1 >= 0 and x2 >= 0 for x1, x2 in P))
if b1_bad:
    fails.append(f"B1: {b1_bad} mismatches")

# B2: random lifts with an area that is not P
b2_max = 0.0
for trial in range(40):
    nD, nT, nO = int(rng.integers(1, 4)), int(rng.integers(1, 4)), int(rng.integers(1, 3))
    nin, nout = int(rng.integers(5, 10)), int(rng.integers(3, 8))
    # layer l(u): units D of u (on in R_u, shelter constant -1 outside), judgment k (1 in R_u, 0 outside), others O
    zD = np.vstack([rng.uniform(0.2, 3, (nin, nD)), -np.ones((nout, nD))])
    hK = np.concatenate([np.ones(nin), np.zeros(nout)])
    zO = np.vstack([-np.ones((nin, nO)), rng.normal(0, 1, (nout, nO))])
    hD, hO = np.maximum(zD, 0), np.maximum(zO, 0)
    WD, WK, WO = rng.normal(0, 1, (nT, nD)), rng.normal(0, 1, nT), np.zeros((nT, nO))  # T reads nothing of O (shelter)
    bT = rng.normal(0, 1, nT) - 5
    zT = hD @ WD.T + np.outer(hK, WK) + hO @ WO.T + bT
    for _ in range(10):
        m, beta = rng.normal(0, 1, nT), rng.normal(0, 2)
        f = np.maximum(zT @ m + beta, 0)
        mhat, betahat = WD.T @ m, beta + m @ (WK * 1.0 + bT)           # kappa = 1 on R_u
        g = np.maximum(zD @ mhat + betahat, 0)
        c0, c1 = f[nin], g[nin]
        pred = g + (c0 - c1) * (1 - hK)
        b2_max = max(b2_max, float(np.abs(f - pred).max()))
        if np.ptp(f[nin:]) > 1e-12 or np.ptp(g[nin:]) > 1e-12:
            fails.append(f"B2-{trial}: not constant off the area")
out["B2"] = dict(trials=400, max_abs_error=b2_max)
if b2_max > 1e-9:
    fails.append(f"B2: max error {b2_max}")

out["failures"] = fails
print(json.dumps(out, indent=1, default=str))
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "logs"), exist_ok=True)
with open(os.path.join(os.path.dirname(__file__), "..", "logs", "tl_netread.json"), "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print("FAILURES:", len(fails))
