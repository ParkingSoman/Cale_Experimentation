# serves: TI3
# serves: DI1
"""Session 1 test of ROADMAP.md (L-sep): may one box unit serve two separated sub-capabilities?
Instance: 5 classes, blocks K1 = {0,1}, K2 = {2,3}, class 4 rest. Group with two nodes (areas) of 4 points each,
weight 1/8 per point. On node w: y_x = ybar_w + a c(x) r_w, c in {-1.5,-.5,.5,1.5}, r1 = (e0-e1)/sqrt2,
r2 = (e2-e3)/sqrt2, ybar_1 = m (1,1,0,0,0), ybar_2 = m (0,0,1,1,0). m = confidence of the area in its own block.
Each node has its own free logit offset t_w (kept judgment units, TH1). Budget k = 1: one output direction q.
For a fixed q the node term is convex (TI3 part 2):
  on   : logits t_w + beta_w c(x) q   (box on throughout, its own read per node)   -> an achievable build
  free : logits t_w + alpha_x q       (any function of the point: a lower bound for that q, switching included)
Outer: q over the unit sphere of the centred space (dim 4), Fibonacci-type seeds + Nelder-Mead refinement.
Pure = q in {r1, r2}. Part 2 (COMMON offset: the judgment unit is inside the group, so one t for both areas and
the unit must also carry the between-area shift; on: alpha = beta_w c + gamma_w per node). Reports pure least, shared least, the q found, and the inactive mass eps (prob outside K_w).
A 'fail' is only a numerical inconsistency (free > on, or a negative KL); the pure-vs-shared outcome is the result."""
import json
import os

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(3132)
C = 5
cvals = np.array([-1.5, -.5, .5, 1.5])
p = np.full(4, 1 / 8)
r1 = np.zeros(C); r1[0], r1[1] = 1, -1; r1 /= np.sqrt(2)
r2 = np.zeros(C); r2[2], r2[3] = 1, -1; r2 /= np.sqrt(2)
# orthonormal basis of the centred space
Bc = np.linalg.qr(np.eye(C) - 1 / C)[0][:, :C - 1]
fails = []


def targets(a, m):
    S = []
    for w, (r, blk) in enumerate([(r1, [0, 1]), (r2, [2, 3])]):
        yb = np.zeros(C); yb[blk] = m
        Y = yb + a * cvals[:, None] * r
        S.append(softmax(Y, axis=1))
    return S


def kl_rows(s, logits):
    lq = logits - logsumexp(logits, axis=1, keepdims=True)
    return np.sum(s * (np.log(s) - lq), axis=1)


def node_term(s, q, mode):
    """least over (t, beta) [on] or (t, alpha) [free] of sum p KL(s || softmax(t + .. q)); convex."""
    n = 1 if mode == "on" else 4

    def f(v):
        t, b = v[:C], v[C:]
        amp = b[0] * cvals if mode == "on" else b
        L = t + amp[:, None] * q
        lq = L - logsumexp(L, axis=1, keepdims=True)
        val = np.sum(p * np.sum(s * (np.log(s) - lq), axis=1))
        g_log = p[:, None] * (np.exp(lq) - s)            # d/dL
        gt = g_log.sum(0)
        gamp = g_log @ q
        gb = np.array([np.sum(gamp * cvals)]) if mode == "on" else gamp
        return val, np.concatenate([gt, gb])

    v0 = np.concatenate([np.log(np.average(s, axis=0, weights=p)), np.zeros(n)])
    r = minimize(f, v0, jac=True, method="L-BFGS-B", options={"maxiter": 2000, "gtol": 1e-11, "ftol": 1e-15})
    return r.fun


def total(S, q, mode):
    q = q / np.linalg.norm(q)
    return sum(node_term(s, q, mode) for s in S)


def info(s):
    sb = np.average(s, axis=0, weights=p)
    return float(np.sum(p * np.sum(s * (np.log(s) - np.log(sb)), axis=1)))


def search(S, mode, nseed=600):
    seeds = rng.normal(size=(nseed, C - 1))
    seeds /= np.linalg.norm(seeds, axis=1, keepdims=True)
    # always include the pure and the equal-shared directions
    extra = [r1, r2, (r1 + r2) / np.sqrt(2), (r1 - r2) / np.sqrt(2)]
    cand = [Bc @ u for u in seeds] + extra
    vals = [total(S, q, mode) for q in cand]
    best = np.argsort(vals)[:6]
    out = (np.inf, None)
    for i in best:
        u0 = Bc.T @ cand[i]
        r = minimize(lambda u: total(S, Bc @ u, mode), u0, method="Nelder-Mead",
                     options={"xatol": 1e-7, "fatol": 1e-12, "maxiter": 3000})
        if r.fun < out[0]:
            out = (r.fun, Bc @ (r.x / np.linalg.norm(r.x)))
    return out


rows = []
for m in [0.0, 1.0, 3.0, 6.0]:
    for a in [0.5, 1.0, 2.0]:
        S = targets(a, m)
        eps = [float(np.mean(1 - s[:, blk].sum(1))) for s, blk in zip(S, [[0, 1], [2, 3]])]
        I = [info(s) for s in S]
        row = dict(m=m, a=a, eps=eps, I=I)
        for mode in ["on", "free"]:
            pure = min(total(S, r1, mode), total(S, r2, mode))
            best, q = search(S, mode)
            row[mode] = dict(pure=pure, shared=best, q=np.round(q, 3).tolist(),
                             ratio=best / pure if pure > 0 else None)
        if row["free"]["shared"] > row["on"]["shared"] + 1e-7 or row["free"]["pure"] > row["on"]["pure"] + 1e-7:
            fails.append(f"free > on at m={m}, a={a}")
        if min(row["on"]["shared"], row["free"]["shared"]) < -1e-9:
            fails.append(f"negative KL at m={m}, a={a}")
        rows.append(row)
        print(f"m={m:3.1f} a={a:3.1f} eps={eps[0]:.3f} I=({I[0]:.4f},{I[1]:.4f}) | "
              f"on: pure {row['on']['pure']:.5f} shared {row['on']['shared']:.5f} | "
              f"free: pure {row['free']['pure']:.5f} shared {row['free']['shared']:.5f} | q_on {row['on']['q']}")


def total_common(S, q, mode):
    q = q / np.linalg.norm(q)
    n = 4 if mode == "on" else 8

    def f(v):
        t, b = v[:C], v[C:]
        val, gt, gb = 0.0, np.zeros(C), np.zeros(n)
        for w, s in enumerate(S):
            amp = (b[2 * w] * cvals + b[2 * w + 1]) if mode == "on" else b[4 * w:4 * w + 4]
            L = t + amp[:, None] * q
            lq = L - logsumexp(L, axis=1, keepdims=True)
            val += np.sum(p * np.sum(s * (np.log(s) - lq), axis=1))
            g = p[:, None] * (np.exp(lq) - s)
            gt += g.sum(0)
            ga = g @ q
            if mode == "on":
                gb[2 * w] += np.sum(ga * cvals); gb[2 * w + 1] += ga.sum()
            else:
                gb[4 * w:4 * w + 4] += ga
        return val, np.concatenate([gt, gb])

    sb = sum(np.average(s, axis=0, weights=p) for s in S) / 2
    best = np.inf
    for _ in range(3):
        v0 = np.concatenate([np.log(sb), rng.normal(scale=2, size=n)])
        r = minimize(f, v0, jac=True, method="L-BFGS-B", options={"maxiter": 4000, "gtol": 1e-11, "ftol": 1e-15})
        best = min(best, r.fun)
    return best


def search_common(S, mode, nseed=300):
    seeds = rng.normal(size=(nseed, C - 1)); seeds /= np.linalg.norm(seeds, axis=1, keepdims=True)
    cand = [Bc @ u for u in seeds] + [r1, r2, (r1 + r2) / np.sqrt(2)]
    vals = [total_common(S, q, mode) for q in cand]
    out = (np.inf, None)
    for i in np.argsort(vals)[:4]:
        r = minimize(lambda u: total_common(S, Bc @ u, mode), Bc.T @ cand[i], method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-11, "maxiter": 1500})
        if r.fun < out[0]:
            out = (r.fun, Bc @ (r.x / np.linalg.norm(r.x)))
    return out


rows2 = []
print("Part 2: common offset (judgment unit inside the group), k = 1")
for m in [0.0, 3.0, 6.0]:
    a = 1.0
    S = targets(a, m)
    I_all = info(np.vstack(S)) if False else None
    row = dict(m=m, a=a)
    for mode in ["on"]:
        pure = min(total_common(S, r1, mode), total_common(S, r2, mode))
        best, q = search_common(S, mode)
        row[mode] = dict(pure=pure, shared=best, q=np.round(q, 3).tolist())
    rows2.append(row)
    print(f"m={m:3.1f} a={a} | common offset, on: pure {row['on']['pure']:.5f} best {row['on']['shared']:.5f} q {row['on']['q']}")

print("fails:", len(fails), fails)
json.dump(dict(rows=rows, rows_common=rows2, fails=fails), open(os.path.join(HERE, "ts_sep.json"), "w"), indent=1)
