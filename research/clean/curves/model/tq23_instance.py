# serves: TQ23
"""Exact check of Lemmas TQ23.a, TQ23.b and the reduction of Theorem TQ23, on small N(M, W; a, B). Float64.
KL is computed in a stable form, from d = (logits) - y: KL = log1p(x) - x + sum s phi(d), x = sum s expm1(d), phi(d) = expm1(d) - d, after shifting d so that d[b_w] = 0. Every value is divided by Gamma.
Corrected after the first full run, which reported 3 fails: that test compared a fit from random starts with the upper value, and a fit can stall. The upper side is a construction, so part A now evaluates the proof's own build directly and starts one fit there. A fit can refute only the lower side.
Estimator, fixed before the run. A: for several M, W (d in 1, 10, 100; biadjacency and random M), p*_1 by a fine grid over unit v on the tail with closed-form u, then polished; p*_r by alternating least squares from many starts. Part 1 fit: V, codes, t_w from random starts and from the formula's build, k = 1, 2. Compare fit / Gamma to p*: it must lie in [(1 - 4 alpha - 6 eta) p*, (1 + 3 alpha) p*]. Fits are upper values, so a fit can refute the lower side and not prove it. B: every claim of Definition DQ18 at eps = 0, from the paper's formulas. C: Part 2, T empty, k = m - 1 + r: at B = 12, 25, 50, fit from random and structured starts, against the same interval, and the proved lower side with nu. D: the reduction: for three graphs at d = 2^14 |E|^6, biclique value by enumeration against p*_1 and against fit / Gamma from the biclique start and random starts.
Prediction, written before the run: A: all fits inside the interval, at ratio about 1 within 4 alpha. C: at B = 12 the fit may sit below p* because a heavy direction can serve the tail; the fit / Gamma approaches p* as B grows; it is at p* within 1% at B = 50. D: |E| - |E*| - 1/4 < p*_1 <= |E| - |E*| and fit / Gamma within 1/8 of it. E: padding: every value inside the interval of Lemma TQ23.c. F: C8 has rank 3 and one exact copy beats three bicliques.
Usage: tq23_instance.py [--smoke]"""
import argparse
import itertools
import json
import math
import os

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
STARTS = 3 if args.smoke else 6
ITERS = 300 if args.smoke else 1500
rng = np.random.default_rng(23)


def build(M, W, a, B):
    M, W = torch.as_tensor(M, dtype=torch.float64), torch.as_tensor(W, dtype=torch.float64)
    m, n = M.shape
    C = 1 + m + n
    y0 = torch.zeros(m, C)
    y0[:, 0] = B - 1
    y0[torch.arange(m), 1 + torch.arange(m)] = B
    y0[:, 1 + m:] = W.log()
    mv = torch.zeros(m, C)
    mv[:, 1 + m:] = M
    d = float(W.max())
    E = float((W * M ** 2).sum())
    tau = 1 / (math.exp(B) * (1 + math.exp(-1)))
    eta = (m + 2 * n * d) * math.exp(-B)
    return dict(M=M, W=W, a=a, B=B, m=m, n=n, C=C, y0=y0, mv=mv, d=d, E=E, tau=tau, eta=eta, alpha=a * math.sqrt(E), Gamma=a ** 2 * tau / (8 * m))


def kl_stable(y, d, piv):
    """KL(softmax(y) || softmax(y + d)), rows; piv: index of each row's pivot class."""
    d = d - d.gather(1, piv[:, None])
    s = torch.softmax(y, 1)
    em = torch.expm1(d)
    phi = torch.where(d.abs() < 1e-3, d ** 2 / 2 + d ** 3 / 6 + d ** 4 / 24, em - d)
    x = (s * em).sum(1)
    l1 = torch.where(x.abs() < 1e-3, -x ** 2 / 2 + x ** 3 / 3 - x ** 4 / 4, torch.log1p(x) - x)
    return l1 + (s * phi).sum(1)


def loss(inst, lp):
    """lp: 2m x C logits of a build; rows ordered (x_w^+, x_w^-). Returns D / Gamma."""
    m = inst["m"]
    y = torch.cat([inst["y0"] + inst["a"] / 2 * inst["mv"], inst["y0"] - inst["a"] / 2 * inst["mv"]])
    piv = (1 + torch.arange(m)).repeat(2)
    return kl_stable(y, lp - y, piv).sum() / (2 * m) / inst["Gamma"]


def pstar(M, W, r, starts=200):
    M, W = np.asarray(M, float), np.asarray(W, float)
    m, n = M.shape
    best, arg = np.inf, None
    for _ in range(starts):
        V = rng.normal(size=(n, r))
        for _ in range(400):
            U = np.stack([np.linalg.lstsq(V * np.sqrt(W[i])[:, None], M[i] * np.sqrt(W[i]), rcond=None)[0] for i in range(m)])
            V = np.stack([np.linalg.lstsq(U * np.sqrt(W[:, j])[:, None], M[:, j] * np.sqrt(W[:, j]), rcond=None)[0] for j in range(n)])
        val = float((W * (M - U @ V.T) ** 2).sum())
        if val < best:
            best, arg = val, (U.copy(), V.copy())
    return best, arg


def build_value(inst, X):
    """The build of the upper side of Lemma TQ23.a: logits y0 +- (a/2) X on the tail. Returns D / Gamma."""
    m, C = inst["m"], inst["C"]
    Xf = torch.zeros(m, C)
    Xf[:, 1 + m:] = torch.as_tensor(X, dtype=torch.float64)
    a = inst["a"]
    return float(loss(inst, torch.cat([inst["y0"] + a / 2 * Xf, inst["y0"] - a / 2 * Xf])))


def lbfgs(ps, fn, it):
    o = torch.optim.LBFGS(ps, lr=1, max_iter=it, history_size=50, line_search_fn="strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-14)

    def cl():
        o.zero_grad()
        v = fn()
        v.backward()
        return v
    o.step(cl)
    return float(fn().detach())


def fit_part1(inst, r, X0=None):
    """areas kept: logits t_w + V c(x), V C x r, codes free, per point."""
    m, C = inst["m"], inst["C"]
    best = np.inf
    for s_ in range(STARTS + (X0 is not None)):
        V = torch.randn(C, r) * .1
        c = torch.randn(2 * m, r) * inst["a"]
        if s_ == 0 and X0 is not None:
            U, v = X0
            V = torch.zeros(C, r)
            V[1 + m:] = torch.as_tensor(v, dtype=torch.float64).reshape(-1, r)
            u = torch.as_tensor(U, dtype=torch.float64).reshape(m, r)
            c = torch.cat([inst["a"] / 2 * u, -inst["a"] / 2 * u])
        V = V.clone().requires_grad_(True)
        c = c.clone().requires_grad_(True)
        t = inst["y0"].clone().requires_grad_(True)
        f = lambda: loss(inst, t.repeat(2, 1) + c @ V.T)
        best = min(best, lbfgs([V, c, t], f, ITERS))
    return best


def fit_part2(inst, r, X0=None):
    """no kept units: logits t + V c_x, V C x (m - 1 + r), codes free per point."""
    m, C = inst["m"], inst["C"]
    k = m - 1 + r
    y0 = inst["y0"]
    best = np.inf
    for s_ in range(STARTS + (X0 is not None)):
        Vb = (y0[1:] - y0[0]).T.clone()
        ex = torch.randn(C, r) * .1
        cb = torch.zeros(2 * m, m - 1)
        cb[torch.arange(1, m) + 0, torch.arange(m - 1)] = 1
        cb[torch.arange(1, m) + m, torch.arange(m - 1)] = 1
        ce = torch.randn(2 * m, r) * inst["a"]
        if s_ == 0 and X0 is not None:
            U, v = X0
            ex = torch.zeros(C, r)
            ex[1 + m:] = torch.as_tensor(v, dtype=torch.float64).reshape(-1, r)
            u = torch.as_tensor(U, dtype=torch.float64).reshape(m, r)
            ce = torch.cat([inst["a"] / 2 * u, -inst["a"] / 2 * u])
        V = torch.cat([Vb, ex], 1).requires_grad_(True)
        c = torch.cat([cb, ce], 1).requires_grad_(True)
        t = y0[0].clone().requires_grad_(True)
        f = lambda: loss(inst, t + c @ V.T)
        best = min(best, lbfgs([V, c, t], f, ITERS))
    return best


def rank1_factors(M, W, starts=200):
    M, W = np.asarray(M, float), np.asarray(W, float)
    m, n = M.shape
    best, arg = np.inf, None
    for _ in range(starts):
        v = rng.normal(size=n)
        for _ in range(400):
            u = (W * M * v).sum(1) / np.maximum((W * v ** 2).sum(1), 1e-300)
            v = (W * M * u[:, None]).sum(0) / np.maximum((W * u[:, None] ** 2).sum(0), 1e-300)
        val = float((W * (M - np.outer(u, v)) ** 2).sum())
        if val < best:
            best, arg = val, (u.copy(), v.copy())
    return best, arg


def biclique(M):
    M = np.asarray(M)
    m, n = M.shape
    best = 0
    for bits in itertools.product([0, 1], repeat=m):
        rows = [i for i in range(m) if bits[i]]
        if not rows:
            continue
        cols = [j for j in range(n) if all(M[i, j] for i in rows)]
        best = max(best, len(rows) * len(cols))
    return best


out = dict(A=[], B=[], C=[], D=[], E=[])
G3 = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1]], float)
C5 = np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1], [1, 0, 0, 0, 1]], float)
fails = 0
# A
cases = [("G3", G3, 1.), ("G3", G3, 10.), ("G3", G3, 100.)]
if not args.smoke:
    Mr = rng.uniform(0, 1, size=(3, 4))
    Wr = np.exp(rng.uniform(0, np.log(30), size=(3, 4)))
    cases += [("rand", Mr, Wr), ("C5", C5, 10.)]
for name, M, dW in cases:
    W = (M + dW * (1 - M)) if np.isscalar(dW) else dW
    W = np.maximum(W, 1.)
    for r in ((1,) if args.smoke else (1, 2)):
        a = 1 / (9 * math.sqrt((W * M ** 2).sum())) / 2
        inst = build(M, W, a, 14.)
        p, (Us, Vs) = pstar(M, W, r, 60 if args.smoke else 200)
        val_build = build_value(inst, Us @ Vs.T)
        v = fit_part1(inst, r, (Us, Vs))
        lo, hi = (1 - 4 * inst["alpha"] - 6 * inst["eta"]) * p, (1 + 3 * inst["alpha"]) * p
        ok = lo - 1e-6 <= v and val_build <= hi + 1e-6
        fails += not ok
        row = dict(name=name, d=inst["d"], r=r, alpha=inst["alpha"], eta=inst["eta"], pstar=p, build_over_Gamma=val_build, fit_over_Gamma=v, lo=lo, hi=hi, ok=bool(ok))
        out["A"].append(row)
        print("A", row, flush=True)
# B: every claim at eps = 0 on a built network
M, W = G3, G3 + 10 * (1 - G3)
m, n = M.shape
h = np.zeros((2 * m, m))
zs = np.zeros((2 * m, m))
for w in range(m):
    h[w, w] = 1
    h[w + m, w] = 1
    zs[w, w] = 3
    zs[w + m, w] = 1
p = np.ones(2 * m) / (2 * m)
inst = build(M, W, .01, 14.)
OUTW = np.zeros((m, inst["C"]))  # logit weight of h_w
OUTS = (inst["a"] / 2) * inst["mv"].numpy()
OUTW = (inst["y0"] - inst["a"] * inst["mv"]).numpy()
Y = h @ OUTW + zs @ OUTS
Yt = np.concatenate([(inst["y0"] + inst["a"] / 2 * inst["mv"]).numpy(), (inst["y0"] - inst["a"] / 2 * inst["mv"]).numpy()])
e_net = float(np.abs(Y - Yt).max())
area = lambda w: [w, w + m]
e_dist = max(float(sum(p[x] * h[x, u] for x in range(2 * m) if x not in area(u))) for u in range(m))
e_dist2 = max(float(sum(p[x] * zs[x, u] for x in range(2 * m) if x not in area(u))) for u in range(m))
e_judg = max(float(sum(p[x] * (1 - h[x, w]) ** 2 for x in area(w))) for w in range(m))
e_on = max(float(np.ptp(h[area(w), w])) for w in range(m))
e_lin = 0.0
for w in range(m):
    rows = area(w)
    Yp = Yt[rows] - Yt[rows].mean(1, keepdims=True)
    z_ = zs[rows, w]
    zc = z_ - z_.mean()
    Yc = Yp - Yp.mean(0)
    Bw = (Yc * zc[:, None]).sum(0) / (zc ** 2).sum()
    e_lin = max(e_lin, float(((Yc - zc[:, None] * Bw[None, :]) ** 2).sum()))
row = dict(net_vs_table=e_net, district=max(e_dist, e_dist2), judgment=e_judg, on_inside=e_on, linear_read=e_lin)
out["B"].append(row)
print("B", row, flush=True)
fails += max(row.values()) > 1e-12
# C: no kept units
for name, M, dW, r in ((("G3", G3, 10., 1),) if args.smoke else (("G3", G3, 10., 1), ("G3", G3, 100., 1), ("C5", C5, 10., 1))):
    W = M + dW * (1 - M)
    a = 1 / (9 * math.sqrt((W * M ** 2).sum())) / 2
    p, (u, v) = rank1_factors(M, W, 100)
    for B in ((14.,) if args.smoke else (12., 25., 50., 200.)):
        inst = build(M, W, a, B)
        val = fit_part2(inst, r, (u, v))
        nu = 64 * inst["m"] * inst["E"] * math.sqrt(inst["m"] * inst["n"] * inst["d"]) * (math.log(inst["d"]) + 2) / B
        lo = (1 - 4 * inst["alpha"] - 6 * inst["eta"]) * (p - nu)
        hi = (1 + 3 * inst["alpha"]) * p
        ok = lo - 1e-6 <= val <= hi + 1e-6
        fails += not ok
        row = dict(name=name, d=inst["d"], B=B, pstar=p, fit_over_Gamma=val, ratio=val / p, nu=nu, lo=lo, hi=hi, ok=bool(ok))
        out["C"].append(row)
        print("C", row, flush=True)
# D: the reduction
for name, M in (("G3", G3),) + ((("C5", C5), ("path", np.array([[1, 1, 0], [0, 1, 1]], float))) if not args.smoke else ()):
    E = int(M.sum())
    Es = biclique(M)
    d = math.exp(math.ceil(14 * math.log(2) + 6 * math.log(E)))
    W = M + d * (1 - M)
    p, (u, v) = rank1_factors(M, W, 200)
    a = 1 / (64 * E ** 2)
    B = math.ceil(math.log(96 * E * (M.shape[0] + 2 * M.shape[1] * d))) + 1
    inst = build(M, W, a, float(B))
    # biclique start
    bestv = np.inf
    for bits in itertools.product([0, 1], repeat=M.shape[0]):
        rows = [i for i in range(M.shape[0]) if bits[i]]
        if not rows:
            continue
        cols = [j for j in range(M.shape[1]) if all(M[i, j] for i in rows)]
        if len(rows) * len(cols) != Es:
            continue
        uu, vv = np.zeros(M.shape[0]), np.zeros(M.shape[1])
        uu[rows] = 1
        vv[cols] = 1
        bestv = min(bestv, fit_part1(inst, 1, (uu, vv)))
        # the same start for p*: alternating steps from a random start get stuck when d is large, so polish from the biclique too
        for _ in range(400):
            vv = (W * M * uu[:, None]).sum(0) / np.maximum((W * uu[:, None] ** 2).sum(0), 1e-300)
            uu = (W * M * vv).sum(1) / np.maximum((W * vv ** 2).sum(1), 1e-300)
        p = min(p, float((W * (M - np.outer(uu, vv)) ** 2).sum()))
    ok = (E - Es - .25 < p <= E - Es + 1e-9) and abs(bestv - (E - Es)) <= 1 / 8 + 1e-6
    fails += not ok
    row = dict(name=name, E=E, Estar=Es, d=d, pstar=p, fit_over_Gamma=bestv, eta=inst["eta"], alpha=inst["alpha"], ok=bool(ok))
    out["D"].append(row)
    print("D", row, flush=True)
# E: Lemma TQ23.c, padding. Values of the weighted fit only, no KL.
for name, M, r in ((("G3", G3, 2),) if args.smoke else (("G3", G3, 2), ("G3", G3, 3), ("path", np.array([[1, 1, 0], [0, 1, 1]], float), 2))):
    m, n = M.shape
    E = int(M.sum())
    Es = biclique(M)
    t = int(math.isqrt(4 * E)) + 1
    d = max(2 ** 14 * E ** 6, 65536 * r ** 2 * E ** 3 * m * n)
    mp_, np_ = m + (r - 1) * t, n + (r - 1) * t
    Mp = np.zeros((mp_, np_))
    Mp[:m, :n] = M
    for l in range(r - 1):
        Mp[m + l * t:m + (l + 1) * t, n + l * t:n + (l + 1) * t] = 1
    Wp = Mp + d * (1 - Mp)
    best = np.inf
    # structured start, then random starts: an upper value of p*_r. It may refute the lower side, and it cannot prove it.
    for s_ in range(40 if not args.smoke else 8):
        V = rng.normal(size=(np_, r)) * .5
        if s_ < 10:
            V = np.zeros((np_, r))
            V[n:, 1:] = 0
            for l in range(r - 1):
                V[n + l * t:n + (l + 1) * t, l + 1] = 1
            V[:n, 0] = rng.integers(0, 2, size=n)
            V = V + 1e-3 * rng.normal(size=V.shape)
        for _ in range(300):
            U = np.stack([np.linalg.lstsq(V * np.sqrt(Wp[i])[:, None], Mp[i] * np.sqrt(Wp[i]), rcond=None)[0] for i in range(mp_)])
            V = np.stack([np.linalg.lstsq(U * np.sqrt(Wp[:, j])[:, None], Mp[:, j] * np.sqrt(Wp[:, j]), rcond=None)[0] for j in range(np_)])
        best = min(best, float((Wp * (Mp - U @ V.T) ** 2).sum()))
    ok = E - Es - 5 / 16 < best and best <= E - Es + 1e-6
    fails += not ok
    row = dict(name=name, r=r, t=t, d=d, E=E, Estar=Es, lower=E - Es - 5 / 16, fit=best, upper=E - Es, ok=bool(ok))
    out["E"].append(row)
    print("E", row, flush=True)
# F: the cycle C8 at r = 3 on three copies: real rank below the biclique count, so the source's sketch of its Remark 3 fails
C8 = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]], float)
if not args.smoke:
    Z_ = np.zeros((4, 4))
    M3 = np.block([[C8, Z_, Z_], [Z_, C8, Z_], [Z_, Z_, C8]])
    W3 = M3 + 1e8 * (1 - M3)
    X3 = np.zeros((12, 12))
    X3[:4, :4] = C8
    val = float((W3 * (M3 - X3) ** 2).sum())
    row = dict(rank_C8=int(np.linalg.matrix_rank(C8)), value_exact_on_one_copy=val, value_three_bicliques=3 * (8 - biclique(C8)))
    out["F"] = row
    print("F", row, flush=True)
    fails += not (row["rank_C8"] == 3 and val < row["value_three_bicliques"])
out["fails"] = int(fails)
print("fails", fails)
json.dump(out, open(os.path.join(HERE, "tq23_instance_smoke.json" if args.smoke else "tq23_instance.json"), "w"), indent=1)
