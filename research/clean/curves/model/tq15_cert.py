# serves: DQ11
"""A certified global optimum for a table where the span has two free angles (Route 1 of Conjecture DQ11). Estimator, fixed before the run.

Instance: K = C = 4, region weights pi_c, table rows sbar_c (all entries > 0). The logits on region c are t + V c_c, with V a C x k matrix. Because t is free, only span(V) mod the all-ones vector matters: a
subspace of dimension k in the 3-dim space P_perp. For k = 2 it is the plane with normal n in S^2. For k = 1 it is the line spanned by n in S^2. So F(n), with n and -n the same span, has two free angles.
For a fixed n, the inner problem is convex, solved to a small tolerance with Newton on the free variables.

Lower bound for a cell: Lemma TQ15 part 3: min of F at the corners minus Cb theta^2 / 2, with Cb from the two-class bound of its part 1. The inner problem is solved to 1e-11.
Branch and bound by quartering cells on three cube faces (a straight line in the chart is a great circle, so a cell is geodesically convex). Output: certified LB, UB, gap. Prune at UB (1 - 1e-3).
Instances: (i) symmetric table a = 0.6, 0.9, k = 2 (pole build value by the formula of DQ11); (ii) symmetric, k = 1; (iii) three random tables; (iv) the small trained net's table: 4 classes picked from the CIFAR net's labels,
s = its softmax restricted and renormalised. Prediction, written before the run: the pole build's value is within the gap of the certified least for k = 2 on the symmetric table.
Usage: tq15_cert.py [--smoke]"""
import argparse
import json
import math
import os
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--maxcells", type=int, default=120000)
ap.add_argument("--only", type=int, default=-1)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
C = 4
E = np.linalg.qr(np.eye(C)[:, :-1] - 1.0 / C)[0]    # orthonormal basis of P_perp, C x 3


def softmax(l):
    l = l - l.max(-1, keepdims=True)
    e = np.exp(l)
    return e / e.sum(-1, keepdims=True)


def basis(n, k):
    """Columns: a basis of the span in R^3 (k = 1: n itself; k = 2: the plane orthogonal to n)."""
    if k == 1:
        return n[:, None]
    Q = np.linalg.svd(n[None, :])[2][1:]
    return Q.T


def inner(S, pi, B, it0=None):
    """min over t (C values) and codes (K x k) of sum pi_c KL(S_c || softmax(t + B c_c)). Alternating Newton on codes and on t: convex, damped; stops on the gradient."""
    K = S.shape[0]
    k = B.shape[1]
    t = np.zeros(C) if it0 is None else it0.copy()
    c = np.zeros((K, k))
    logS = np.log(S)
    def val(t, c):
        l = t + c @ B.T
        l = l - l.max(1, keepdims=True)
        lq = l - np.log(np.exp(l).sum(1, keepdims=True))
        return float((pi * (S * (logS - lq)).sum(1)).sum())
    f = val(t, c)
    for outer in range(200):
        for ci in range(K):
            for _ in range(3):
                q = softmax(t + B @ c[ci])
                g = B.T @ (q - S[ci])
                H = B.T @ (np.diag(q) - np.outer(q, q)) @ B + 1e-10 * np.eye(k)
                st = np.linalg.solve(H, g)
                lam = 1.0
                f0 = (S[ci] * (logS[ci] - np.log(softmax(t + B @ c[ci])))).sum()
                while lam > 1e-6:
                    c2 = c[ci] - lam * st
                    if (S[ci] * (logS[ci] - np.log(softmax(t + B @ c2)))).sum() <= f0 - 1e-4 * lam * g @ st:
                        break
                    lam /= 2
                c[ci] = c[ci] - lam * st
        for _ in range(3):
            q = softmax(t + c @ B.T)
            g = (pi[:, None] * (q - S)).sum(0)
            H = sum(pi[i] * (np.diag(q[i]) - np.outer(q[i], q[i])) for i in range(K)) + 1e-10 * np.eye(C)
            st = np.linalg.solve(H, g)
            lam = 1.0
            f0 = val(t, c)
            while lam > 1e-6 and val(t - lam * st, c) > f0 - 1e-4 * lam * g @ st:
                lam /= 2
            t = t - lam * st
        f2 = val(t, c)
        if abs(f - f2) < 1e-11:
            f = f2
            break
        f = f2
    return f, t


def spread_bound(S, pi, U):
    """Largest |l_i - l_j| over regions and pairs at a build of total loss <= U: two-class KL(p || q) <= U / (pi_c (S_i + S_j)) by the chain rule."""
    worst = 0.0
    K = S.shape[0]
    for c in range(K):
        for i in range(C):
            for j in range(i + 1, C):
                m = S[c, i] + S[c, j]
                p = S[c, i] / m
                cap = U / (pi[c] * m)
                def kl2(d):
                    q = 1 / (1 + math.exp(-d))
                    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
                d0 = math.log(p / (1 - p))
                for sign in (1, -1):
                    lo, hi = 0.0, 1.0
                    while kl2(d0 + sign * hi) < cap and hi < 1e6:
                        hi *= 2
                    for _ in range(60):
                        mid = (lo + hi) / 2
                        if kl2(d0 + sign * mid) < cap:
                            lo = mid
                        else:
                            hi = mid
                    worst = max(worst, abs(d0 + sign * hi))
    return worst


def to_n(face, u, v):
    n = [(1.0, u, v), (u, 1.0, v), (u, v, 1.0)][face]
    n = np.array(n)
    return n / np.linalg.norm(n)


def certify(S, pi, k, name, tol=1e-3, maxcells=400000):
    """Branch and bound by Lemma TQ15 part 3. Cells are squares on three faces of a cube (central projection), which cover every line through 0 once: straight lines of the chart are great circles, so a
    square's image is a geodesically convex polygon and every interior point has its corners' logarithm vectors around 0. theta is the polygon's diameter (attained at corners)."""
    t0 = time.time()
    memo = {}

    def F(face, u, v):
        key = (face, round(u, 12), round(v, 12))
        if key not in memo:
            memo[key] = inner(S, pi, E @ basis(to_n(face, u, v), k))[0]
        return memo[key]

    N0 = 6
    h = 2.0 / N0
    frontier = [(f, -1 + i * h, -1 + (i + 1) * h, -1 + j * h, -1 + (j + 1) * h) for f in range(3) for i in range(N0) for j in range(N0)]
    UB = min(F(f, u, v) for f, u0, u1, v0, v1 in frontier for u in (u0, u1) for v in (v0, v1))
    B = spread_bound(S, pi, UB) * math.sqrt(C)
    Cb = B * B + 2 * B
    LB = 1e9
    used = 0
    rounds = 0

    def lower(c):
        f, u0, u1, v0, v1 = c
        ns = [to_n(f, u, v) for u in (u0, u1) for v in (v0, v1)]
        theta = max(math.acos(min(1.0, abs(float(a_ @ b_)))) for a_ in ns for b_ in ns)
        return min(F(f, u, v) for u in (u0, u1) for v in (v0, v1)) - .5 * Cb * theta ** 2

    while frontier and len(memo) < maxcells:
        nxt = []
        for c in frontier:
            f, u0, u1, v0, v1 = c
            um, vm = (u0 + u1) / 2, (v0 + v1) / 2
            UB = min(UB, F(f, um, vm))
            lb = lower(c)
            used += 1
            if lb >= UB * (1 - tol):
                LB = min(LB, lb)
            else:
                nxt += [(f, u0, um, v0, vm), (f, u0, um, vm, v1), (f, um, u1, v0, vm), (f, um, u1, vm, v1)]
        frontier = nxt
        rounds += 1
        if a.smoke and rounds >= 3:
            break
    if frontier:
        LB = min(LB, min(lower(c) for c in frontier))
    return dict(name=name, k=k, UB=UB, LB=LB, gap=UB - LB, F_evals=len(memo), cells=used, open_cells=len(frontier), Cb=Cb, sec=time.time() - t0, certified_to_tol=not frontier)


def sym(a):
    b = (1 - a) / (C - 1)
    return np.full((C, C), b) + np.eye(C) * (a - b)


def pole_value(a, K):
    b = (1 - a) / (K - 1)
    r = a - b
    A, Bq, Cq = 1 - r, -r * (K - 2), -(1 + r)
    u = (-Bq + math.sqrt(Bq * Bq - 4 * A * Cq)) / (2 * A)
    tau = math.log(u)
    Hs = -a * math.log(a) - (K - 1) * b * math.log(b)
    return math.log(u + 1 / u + K - 2) - r * tau - Hs


rows = []
pi = np.ones(C) / C
if a.smoke:
    insts = [("sym a=.9", sym(.9), pi, 2)]
else:
    rng = np.random.default_rng(5)
    insts = [("sym a=.6", sym(.6), pi, 2), ("sym a=.9", sym(.9), pi, 2), ("sym a=.9", sym(.9), pi, 1)]
    for r in range(3):
        y = rng.normal(size=(C, C)) * 2
        insts.append((f"random {r}", softmax(y), pi, 2))
    p = os.path.join(HERE, "..", "cache", "map", "cifar_net.npz")
    if os.path.exists(p):
        d = np.load(p)
        l = d["h"] @ d["W"].T + d["b"]
        cl = [3, 42, 55, 88]
        s = softmax(l)[:, cl]
        s = s / s.sum(1, keepdims=True)
        tab = np.stack([s[d["lab"] == c].mean(0) for c in cl])
        tab = np.maximum(tab, 1e-6)
        tab = tab / tab.sum(1, keepdims=True)
        insts.append(("cifar net, 4 classes", tab, pi, 2))
        insts.append(("cifar net, 4 classes", tab, pi, 1))
for i_, (name, S, w, k) in enumerate(insts):
    if a.only >= 0 and i_ != a.only:
        continue
    r = certify(S, w, k, name, maxcells=20000 if a.smoke else a.maxcells)
    if name.startswith("sym") and k == 2:
        r["pole_formula"] = pole_value(float(S[0, 0]), C)
        r["pole_above_UB"] = r["pole_formula"] - r["UB"]
    rows.append(r)
    print({x: (round(v, 6) if isinstance(v, float) else v) for x, v in r.items() if x != "best"}, flush=True)
if not a.smoke:
    json.dump(rows, open(os.path.join(HERE, f"tq15_cert_{a.only}.json"), "w"), indent=1)
print("done", flush=True)
