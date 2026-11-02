# serves: DQ2
"""Check for Conjecture DQ2 beyond two areas (COARSE-THEORY.md D.18), in the second-order leak model of its Route
step (2): n areas, each with own coordinates of strengths S_w (k x k, positive definite) and, for each other area
w', the metric M_(w,w') (k x k, positive semidefinite) of block K_w' on area w in w''s own coordinates. One
invertible k x k map A_w per area (A_1 = I: the cost is unchanged by A_w -> A_w G for one common G).
   cost(A) = sum over ordered pairs w != w' of tr(M_(w,w') T S_w T^T),   T = A_w' A_w^-1.
Questions (each instance):
 (a) DIAGONAL instances (S_w, M diagonal): is the least over GL(k)^(n-1) (many restarts) equal to the least over
     permutations + diagonal scales? With A_w = P_w D_w the cost splits by unit, and for fixed permutations the
     scales are a convex problem in log-coordinates (a sum of exponentials of linear forms): the permutation
     problem is an n-dimensional axial assignment with costs c(i_1, ..., i_n) = min over scales.
 (b) For n = 2 this is Lemma TQ4 (reverse pairing). For n = 3: is the best assignment "pairwise reverse"?
 (c) GENERAL instances (S, M not diagonal): the GL optimum against the best of (permutation + scale) in the own
     eigenbases, to see whether mixing helps.
Exact enumeration of all assignments (k <= 3, n <= 4); the convex scale problem solved to 1e-12."""
import itertools
import json
import os

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20261101)


def cost_A(As, S, M):
    n = len(As); tot = 0.0
    for w in range(n):
        Ainv = np.linalg.inv(As[w])
        for w2 in range(n):
            if w2 != w:
                Tm = As[w2] @ Ainv
                tot += np.trace(M[w][w2] @ Tm @ S[w] @ Tm.T)
    return tot


def cost_grad(As, S, M):
    n = len(As); tot = 0.0; G = [np.zeros_like(A) for A in As]
    Bs = [np.linalg.inv(A) for A in As]
    for w in range(n):
        B = Bs[w]
        for w2 in range(n):
            if w2 == w:
                continue
            A2 = As[w2]; Mm = M[w][w2]
            X = A2 @ B
            tot += np.trace(Mm @ X @ S[w] @ X.T)
            G[w2] += 2 * Mm @ X @ S[w] @ B.T
            G[w] += -2 * B.T @ A2.T @ Mm @ X @ S[w] @ B.T
    return tot, G


def gl_min(S, M, k, restarts=12):
    n = len(S); best = np.inf
    def f(x):
        As = [np.eye(k)] + [x[i * k * k:(i + 1) * k * k].reshape(k, k) for i in range(n - 1)]
        try:
            v, G = cost_grad(As, S, M)
        except np.linalg.LinAlgError:
            return 1e300, np.zeros_like(x)
        return v, np.concatenate([g.ravel() for g in G[1:]])
    for _ in range(restarts):
        x0 = rng.normal(size=(n - 1) * k * k)
        r = minimize(f, x0, jac=True, method="BFGS", options=dict(gtol=1e-11, maxiter=50000))
        if r.fun < best:
            best = r.fun
    return best


def unit_cost(a):
    """min over log-scales u (u_0 = 0) of sum_{w != w'} a[w, w'] exp(u_w' - u_w): convex."""
    n = a.shape[0]
    def f(u):
        uu = np.concatenate([[0.0], u]); E = np.exp(uu[None, :] - uu[:, None])
        return float((a * E).sum())
    def g(u):
        uu = np.concatenate([[0.0], u]); E = a * np.exp(uu[None, :] - uu[:, None])
        grad = E.sum(0) - E.sum(1)
        return grad[1:]
    r = minimize(f, np.zeros(n - 1), jac=g, method="BFGS", options=dict(gtol=1e-13, maxiter=10000))
    return r.fun


def assign_min(s, m, k):
    """s[w][i]: strength of own coordinate i of area w; m[w][w2][j]: metric of block w2's coordinate j on area w.
    Enumerate permutations pi_w (pi_0 = identity): unit u carries coordinate pi_w(u) of each area."""
    n = len(s); best = np.inf; bestp = None
    perms = list(itertools.permutations(range(k)))
    for ps in itertools.product(perms, repeat=n - 1):
        pis = [tuple(range(k))] + list(ps); tot = 0.0
        for u in range(k):
            a = np.zeros((n, n))
            for w in range(n):
                for w2 in range(n):
                    if w2 != w:
                        a[w, w2] = m[w][w2][pis[w2][u]] * s[w][pis[w][u]]
            tot += unit_cost(a)
        if tot < best:
            best, bestp = tot, pis
    return best, bestp


fails = []; rows = []
# (a), (b): diagonal instances
for n, k in [(2, 2), (2, 3), (3, 2), (3, 3), (4, 2)]:
    for trial in range(6):
        s = [np.sort(rng.uniform(0.1, 3, k))[::-1] for _ in range(n)]
        m = [[rng.uniform(0.05, 2, k) for _ in range(n)] for _ in range(n)]
        S = [np.diag(x) for x in s]; M = [[np.diag(m[w][w2]) for w2 in range(n)] for w in range(n)]
        ga = gl_min(S, M, k)
        am, ap = assign_min(s, m, k)
        rel = (am - ga) / am
        # pairwise reverse: for n = 2 compare with TQ4; for n >= 3 report whether every pair's induced pairing is
        # the reverse of the order of the products p, q of TQ4 for that pair
        if n == 2:
            p = s[0] * m[1][0]; q = s[1] * m[0][1]
            tq4 = 2 * np.sum(np.sqrt(np.sort(p)[::-1] * np.sort(q)))
            if abs(tq4 - am) > 1e-8 * am:
                fails.append(("tq4", n, k, trial, tq4, am))
        if ga < am * (1 - 1e-6):
            mixing = True
        else:
            mixing = False
        if ga > am * (1 + 1e-6):
            fails.append(("gl_above_assign", n, k, trial, ga, am))   # GL contains the assignments
        rows.append(dict(kind="diag", n=n, k=k, trial=trial, gl=ga, assign=am, gl_below_assign=mixing, rel=rel,
                         perm=[list(p) for p in ap]))
# (c): general instances
for n, k in [(2, 2), (3, 2)]:
    for trial in range(6):
        S = []; M = [[None] * n for _ in range(n)]
        for w in range(n):
            B = rng.normal(size=(k, k)); S.append(B @ B.T + 0.1 * np.eye(k))
            for w2 in range(n):
                B = rng.normal(size=(k, k)); M[w][w2] = B @ B.T + 0.05 * np.eye(k)
        ga = gl_min(S, M, k)
        rows.append(dict(kind="general", n=n, k=k, trial=trial, gl=ga))
summary = {}
for r in rows:
    if r["kind"] == "diag":
        key = f"n{r['n']}k{r['k']}"
        summary.setdefault(key, []).append(r["gl_below_assign"])
print("fails:", len(fails), fails[:5])
print("diagonal: does a non-permutation GL map beat the best assignment? (per (n,k), per trial)")
for key, v in summary.items():
    print(key, v)
json.dump(dict(fails=fails, rows=rows), open(os.path.join(HERE, "tq_ngauge.json"), "w"), indent=1, default=float)
