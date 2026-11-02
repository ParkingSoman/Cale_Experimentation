# serves: T21
"""The exact rule of Theorem T21 (COARSE-THEORY.md B.T), for small instances.

Two logits, one hidden layer, width k, finite mix (X, pi), teacher gap t. Objective 'kl' (D_P) or 'd2' (D2_P).
The least value is the minimum over sign choices s in {-1, 1}^k and region patterns A_1..A_k of a convex problem:
minimise F(sum_j s_j E_{A_j} theta_j + gamma 1) subject to theta_j in Theta_{A_j}. Regions are found exactly by one
linear-programming feasibility test per sign vector (fine for n <= 12). Each convex piece is solved with SLSQP from a
few starts, so values are accurate to the solver tolerance, not exact.
"""
import itertools

import numpy as np
from scipy.optimize import linprog, minimize


def sig(z):
    return 1 / (1 + np.exp(-z))


def per_point_loss(kind, t):
    q = sig(t)
    if kind == "kl":
        H = -(q * np.log(q) + (1 - q) * np.log(1 - q))
        return (lambda f: np.logaddexp(0, f) - q * f - H), (lambda f: sig(f) - q)
    return (lambda f: (f - t) ** 2 / 4), (lambda f: (f - t) / 2)


def regions(X):
    """Every set A = {i : x~_i theta > 0} of a full-dimensional region of the arrangement, plus the empty set."""
    n, d = X.shape
    Xt = np.hstack([X, np.ones((n, 1))])
    out = []
    for signs in itertools.product([1, -1], repeat=n):
        sg = np.array(signs, float)
        # find theta with sg_i x~_i theta >= 1 for all i
        res = linprog(np.zeros(d + 1), A_ub=-(sg[:, None] * Xt), b_ub=-np.ones(n), bounds=[(None, None)] * (d + 1),
                      method="highs")
        if res.status == 0:
            out.append(frozenset(np.where(sg > 0)[0]))
    out.append(frozenset())
    return sorted(set(out), key=lambda a: (len(a), sorted(a))), Xt


def solve_piece(Xt, pi, lossf, gradf, choice, starts=4, rng=None):
    """choice = list of (s_j, A_j). Returns (value, theta list, gamma)."""
    n, D = Xt.shape
    k = len(choice)
    masks = [np.array([1.0 if i in A else 0.0 for i in range(n)]) for _, A in choice]
    signs = [s for s, _ in choice]

    def gap(z):
        th = z[:k * D].reshape(k, D)
        g = z[-1]
        return sum(signs[j] * masks[j] * (Xt @ th[j]) for j in range(k)) + g

    def F(z):
        return float(np.sum(pi * lossf(gap(z))))

    def dF(z):
        r = pi * gradf(gap(z))
        th_g = [signs[j] * Xt.T @ (masks[j] * r) for j in range(k)]
        return np.concatenate(th_g + [np.array([r.sum()])])

    cons = []
    for j, (_, A) in enumerate(choice):
        for i in range(n):
            row = np.zeros(k * D + 1)
            row[j * D:(j + 1) * D] = Xt[i] if i in A else -Xt[i]
            cons.append(row)
    C = np.array(cons)
    con = {"type": "ineq", "fun": lambda z: C @ z, "jac": lambda z: C}
    rng = rng or np.random.default_rng(0)
    best = (np.inf, None)
    for _ in range(starts):
        z0 = np.zeros(k * D + 1)
        z0[:-1] = rng.normal(size=k * D) * 0.1
        r = minimize(F, z0, jac=dF, constraints=[con], method="SLSQP", options={"maxiter": 2000, "ftol": 1e-13})
        z = r.x
        if np.all(C @ z >= -1e-7) and r.fun < best[0]:
            best = (r.fun, z)
    return best


def exact_min(X, pi, t, k, kind="kl", starts=4):
    regs, Xt = regions(X)
    lossf, gradf = per_point_loss(kind, np.asarray(t, float))
    best = (np.inf, None, None)
    unit_choices = [(s, A) for A in regs for s in (1, -1)]
    for choice in itertools.combinations_with_replacement(range(len(unit_choices)), k):
        ch = [unit_choices[c] for c in choice]
        v, z = solve_piece(Xt, np.asarray(pi, float), lossf, gradf, ch, starts=starts)
        if v < best[0]:
            best = (v, z, ch)
    return best, Xt


# ---------------- fold mode (Theorem T27) ----------------
def regions_cone(Zeta):
    """Active sets of full-dimensional regions of the central arrangement {u : zeta_j^T u = 0} inside u >= 0."""
    n, D = Zeta.shape
    out = [frozenset()]
    for signs in itertools.product([1, -1], repeat=n):
        sg = np.array(signs, float)
        A_ub = np.vstack([-(sg[:, None] * Zeta), -np.eye(D)])
        b_ub = np.concatenate([-np.ones(n), np.zeros(D)])
        res = linprog(np.zeros(D), A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * D, method="highs")
        if res.status == 0:
            out.append(frozenset(np.where(sg > 0)[0]))
    return sorted(set(out), key=lambda a: (len(a), sorted(a)))


def solve_piece_fold(Zetas, pi, lossf, gradf, choice, starts=4, rng=None):
    """choice = list of (s_j, group index g_j, A_j); unit j is s_j relu(u_j^T zeta_{g_j}), u_j >= 0."""
    k = len(choice)
    dims = [Zetas[g].shape[1] for _, g, _ in choice]
    offs = np.concatenate([[0], np.cumsum(dims)])
    n = Zetas[0].shape[0]
    masks = [np.array([1.0 if i in A else 0.0 for i in range(n)]) for _, _, A in choice]

    def gap(z):
        out = z[-1] * np.ones(n)
        for j, (s, g, _) in enumerate(choice):
            out = out + s * masks[j] * (Zetas[g] @ z[offs[j]:offs[j + 1]])
        return out

    def F(z):
        return float(np.sum(pi * lossf(gap(z))))

    def dF(z):
        r = pi * gradf(gap(z))
        parts = [s * Zetas[g].T @ (masks[j] * r) for j, (s, g, _) in enumerate(choice)]
        return np.concatenate(parts + [np.array([r.sum()])])

    rows = []
    for j, (s, g, A) in enumerate(choice):
        for i in range(n):
            row = np.zeros(offs[-1] + 1)
            row[offs[j]:offs[j + 1]] = Zetas[g][i] if i in A else -Zetas[g][i]
            rows.append(row)
        for c in range(dims[j]):
            row = np.zeros(offs[-1] + 1); row[offs[j] + c] = 1.0; rows.append(row)
    C = np.array(rows)
    con = {"type": "ineq", "fun": lambda z: C @ z, "jac": lambda z: C}
    rng = rng or np.random.default_rng(0)
    best = (np.inf, None)
    for _ in range(starts):
        z0 = np.zeros(offs[-1] + 1)
        z0[:-1] = np.abs(rng.normal(size=offs[-1])) * 0.1
        r = minimize(F, z0, jac=dF, constraints=[con], method="SLSQP", options={"maxiter": 2000, "ftol": 1e-13})
        if np.all(C @ r.x >= -1e-7) and r.fun < best[0]:
            best = (r.fun, r.x)
    return best


def exact_min_fold(Zs, pi, t, k, kind="kl", starts=3):
    """Zs: list of (n_P x s_g) arrays of group pre-activations z_S(x_j). Units are folds of one group each."""
    Zetas = [np.hstack([Z, -np.ones((Z.shape[0], 1))]) for Z in Zs]
    lossf, gradf = per_point_loss(kind, np.asarray(t, float))
    unit_choices = []
    for g, Ze in enumerate(Zetas):
        for A in regions_cone(Ze):
            for s in (1, -1):
                unit_choices.append((s, g, A))
    best = (np.inf, None, None)
    for choice in itertools.combinations_with_replacement(range(len(unit_choices)), k):
        ch = [unit_choices[c] for c in choice]
        v, z = solve_piece_fold(Zetas, np.asarray(pi, float), lossf, gradf, ch, starts=starts)
        if v < best[0]:
            best = (v, z, ch)
    return best
