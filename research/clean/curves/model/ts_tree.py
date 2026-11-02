# serves: TQ1
# serves: TQ2
# serves: DQ1
"""Checks of Part B.Q (COARSE-THEORY.md): Theorem TQ1, Proposition TQ2, Conjecture DQ1.
A  TQ1 part 2 on random instances: 2 to 4 nodes, 8 to 14 classes, disjoint K_w, Y_w(x) = M_w z(x) with z of size
   r <= k (so own-class rank <= k), confidence m on K_w, random weights. The packed choice is built as in the theorem
   (U_w from Pi_w Y_w, V = sum Pi_w U_w, c_w = U_w^T Pi_w Y_w, t_w = ybar_w - beta^w) and its KL terms are computed
   exactly. Fail if terms > sum eps_act + 1e-9, if eps_act < 0, or if closed form (3) < eps_act.
B  TQ2 part 1: eps_act > 0 when Y != 0 and = 0 when Y = 0.
C  TQ2 part 2: on the instance of model/ts_sep.py (a = 1), min over unit directions v of F(v) = f_1 + f_2 (free
   amplitude per point, free offset per node) on a Fibonacci grid of 3000 directions + refinement is > 0.
D  TQ1 part 1 (code check): for a node of own rank 2 next to a rank-1 node, k = 1, the group's relaxed least is
   >= the sum of the nodes' relaxed leasts.
E  DQ1 (reported): on the ts_sep family with growing m, bound sum eps_act against min_v F(v) (a lower bound of L_S
   for k = 1); the ratio should stay bounded if DQ1 holds."""
import json
import os

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(3133)
fails, out = [], {}


def phi(t):
    return np.expm1(t) - t


def eps_act(Y, s, p, K):
    """Definition DQ1 (revised). Y: centred logits on the area (m x C), s softmax, p weights, K bool mask."""
    rhoK = np.linalg.norm(Y[:, K], axis=1)[:, None]
    U = -Y[:, ~K]
    so = s[:, ~K] * p[:, None]
    lo, hi = np.full(U.shape[1], -60.0), np.full(U.shape[1], 200.0)
    for _ in range(120):
        mid = (lo + hi) / 2
        a1, a2 = U - mid + rhoK, U - mid - rhoK
        g = np.where(phi(a1) >= phi(a2), -np.expm1(a1), -np.expm1(a2))
        sl = (so * g).sum(0)
        lo, hi = np.where(sl < 0, mid, lo), np.where(sl < 0, hi, mid)
    beta = (lo + hi) / 2
    a1, a2 = U - beta + rhoK, U - beta - rhoK
    val = float((so * np.maximum(phi(a1), phi(a2))).sum())
    D = float((np.abs(Y[:, ~K]) + rhoK).max())
    mu = float((so).sum())
    return val, beta, 2 * D * mu


def kl_terms(s, p, L):
    lq = L - logsumexp(L, axis=1, keepdims=True)
    return float(np.sum(p * np.sum(s * (np.log(s) - lq), axis=1)))


# ---------- A ----------
nA, worst = 0, 0.0
for trial in range(300):
    C = int(rng.integers(8, 15))
    nn = int(rng.integers(2, 5))
    k = int(rng.integers(1, 4))
    perm = rng.permutation(C)
    sizes = rng.integers(2, 4, size=nn)
    if sizes.sum() > C - 1:
        continue
    Ks, o = [], 0
    for sz in sizes:
        K = np.zeros(C, bool); K[perm[o:o + sz]] = True; Ks.append(K); o += sz
    m = float(rng.choice([0.0, 2.0, 4.0, 8.0]))
    nodes = []
    for w in range(nn):
        r = int(rng.integers(1, k + 1))
        npts = 6
        z = rng.normal(size=(npts, r))
        Mw = rng.normal(size=(C, r)) * rng.uniform(0.3, 1.5)
        Mw[Ks[w]] *= 2.0
        yb = rng.normal(size=C) * 0.5; yb[Ks[w]] += m
        Yraw = z @ Mw.T
        p = rng.uniform(0.5, 1.5, size=npts); p /= p.sum() * nn
        Pc = (yb + Yraw) - (yb + Yraw).mean(1, keepdims=True)
        ybar = np.average(Pc, axis=0, weights=p)
        Yw = Pc - ybar
        s = softmax(yb + Yraw, axis=1)
        nodes.append(dict(K=Ks[w], Y=Yw, ybar=ybar, s=s, p=p))
    # packed choice
    V = np.zeros((C, k))
    for nd in nodes:
        X = nd["Y"] * nd["K"]                        # Pi_w Y_w
        u, sv, vt = np.linalg.svd(X, full_matrices=False)
        rr = int((sv > 1e-9 * max(sv[0], 1e-12)).sum())
        if rr > k:
            fails.append(f"A trial {trial}: own rank {rr} > k {k}")
        Uw = np.zeros((C, k)); Uw[:, :rr] = vt[:rr].T
        nd["U"] = Uw
        V += nd["K"][:, None] * Uw
    tot, bound = 0.0, 0.0
    for nd in nodes:
        e, beta, closed = eps_act(nd["Y"], nd["s"], nd["p"], nd["K"])
        if e < -1e-12:
            fails.append(f"A trial {trial}: eps_act < 0")
        if closed < e - 1e-9:
            fails.append(f"A trial {trial}: closed form {closed} < eps_act {e}")
        c = (nd["Y"] * nd["K"]) @ nd["U"]
        t = nd["ybar"].copy(); t[~nd["K"]] -= beta
        L = t + c @ V.T
        tot += kl_terms(nd["s"], nd["p"], L)
        bound += e
    if tot > bound + 1e-9:
        fails.append(f"A trial {trial}: packed terms {tot} > bound {bound}")
    worst = max(worst, tot / bound if bound > 0 else 0)
    nA += 1
out["A"] = dict(trials=nA, max_terms_over_bound=worst)
print(f"A: {nA} instances, max packed terms / bound = {worst:.3f}")

# ---------- B ----------
Cb = 6; Kb = np.zeros(Cb, bool); Kb[:2] = True
pb = np.full(4, .25)
Y0 = np.zeros((4, Cb)); s0 = softmax(np.tile(np.arange(Cb, dtype=float), (4, 1)), axis=1)
e0 = eps_act(Y0, s0, pb, Kb)[0]
if abs(e0) > 1e-12:
    fails.append(f"B: eps_act {e0} != 0 for Y = 0")
cnt_pos = 0
for _ in range(200):
    Yb = rng.normal(size=(4, Cb)) * rng.uniform(1e-3, 3)
    Yb -= Yb.mean(1, keepdims=True); Yb -= Yb.mean(0)
    sb = softmax(rng.normal(size=Cb) + Yb, axis=1)
    if eps_act(Yb, sb, pb, Kb)[0] <= 0:
        fails.append("B: eps_act <= 0 for Y != 0")
    cnt_pos += 1
out["B"] = dict(zero_case=e0, positive_cases=cnt_pos)
print(f"B: eps_act(Y=0) = {e0:.1e}; {cnt_pos} cases with Y != 0 all > 0")

# ---------- shared pieces for C, D, E (ts_sep instance) ----------
C5 = 5
cvals = np.array([-1.5, -.5, .5, 1.5]); p8 = np.full(4, 1 / 8)
r1 = np.zeros(C5); r1[0], r1[1] = 1, -1; r1 /= np.sqrt(2)
r2 = np.zeros(C5); r2[2], r2[3] = 1, -1; r2 /= np.sqrt(2)
Bc = np.linalg.qr(np.eye(C5) - 1 / C5)[0][:, :C5 - 1]


def ts_targets(a, m):
    S = []
    for r, blk in [(r1, [0, 1]), (r2, [2, 3])]:
        yb = np.zeros(C5); yb[blk] = m
        S.append(softmax(yb + a * cvals[:, None] * r, axis=1))
    return S


def f_free(s, q, p=p8):
    """inf over t, alpha of sum p KL(s || softmax(t + alpha_x q)); convex."""
    npts = len(s)

    def f(v):
        t, al = v[:C5], v[C5:]
        L = t + al[:, None] * q
        lq = L - logsumexp(L, axis=1, keepdims=True)
        g = p[:, None] * (np.exp(lq) - s)
        return float(np.sum(p * np.sum(s * (np.log(s) - lq), axis=1))), np.concatenate([g.sum(0), g @ q])
    v0 = np.concatenate([np.log(np.average(s, axis=0, weights=p)), np.zeros(npts)])
    return minimize(f, v0, jac=True, method="L-BFGS-B", options={"maxiter": 3000, "gtol": 1e-12, "ftol": 1e-16}).fun


def fib_sphere(nv, dim=4):
    x = rng.normal(size=(nv, dim)); return x / np.linalg.norm(x, axis=1, keepdims=True)


def min_F(S, nv=3000):
    U = fib_sphere(nv)
    vals = np.array([sum(f_free(s, Bc @ u) for s in S) for u in U])
    best = np.inf
    for i in np.argsort(vals)[:5]:
        r = minimize(lambda u: sum(f_free(s, Bc @ (u / np.linalg.norm(u))) for s in S), U[i], method="Nelder-Mead",
                     options={"xatol": 1e-8, "fatol": 1e-14, "maxiter": 2000})
        best = min(best, r.fun)
    return best, float(vals.min())


# ---------- C ----------
out["C"] = {}
for m in [0.0, 3.0, 6.0]:
    mF, grid = min_F(ts_targets(1.0, m), nv=1500)
    if not mF > 0:
        fails.append(f"C: min F = {mF} at m = {m}")
    out["C"][m] = dict(min_F=mF, grid_min=grid)
    print(f"C: m = {m}: min over directions of F = {mF:.3e} (grid {grid:.3e}) > 0")

# ---------- D ----------
s_rank2 = softmax(np.array([0, 0, 0, 0, 0.]) + np.outer(cvals, r1) + np.outer(cvals[::-1] ** 2 - 1.25, r2 + r1 / 3), axis=1)
S_D = [s_rank2, ts_targets(1.0, 3.0)[1]]
sumL = sum(min_F([s], nv=600)[0] for s in S_D)
grpL = min_F(S_D, nv=600)[0]
if grpL < sumL - 1e-7:
    fails.append(f"D: group {grpL} < sum of nodes {sumL}")
out["D"] = dict(group=grpL, sum_nodes=sumL)
print(f"D: group relaxed least {grpL:.5f} >= sum of node leasts {sumL:.5f}")

# ---------- E ----------
out["E"] = {}
for m in [3.0, 6.0, 9.0, 12.0]:
    S = ts_targets(1.0, m)
    bnd = 0.0
    for s, blk in zip(S, [[0, 1], [2, 3]]):
        K = np.zeros(C5, bool); K[blk] = True
        Pc = np.log(s) - np.log(s).mean(1, keepdims=True)
        Yw = Pc - np.average(Pc, axis=0, weights=p8)
        bnd += eps_act(Yw, s, p8, K)[0]
    lowF, _ = min_F(S, nv=800)
    out["E"][m] = dict(bound=bnd, minF=lowF, ratio=bnd / lowF)
    print(f"E: m = {m:4.1f}: bound {bnd:.3e}, min F (lower bound of L_S) {lowF:.3e}, ratio {bnd / lowF:.2f}")

print("fails:", len(fails), fails[:5])
out["fails"] = fails
json.dump(out, open(os.path.join(HERE, "ts_tree.json"), "w"), indent=1, default=float)
