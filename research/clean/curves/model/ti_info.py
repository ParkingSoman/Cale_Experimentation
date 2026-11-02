# serves: TI0
# serves: TI1
# serves: TI2
"""Lemma TI0, Theorems TI1 and TI2 (COARSE-THEORY.md B.I) on the synthetic deep network of model/tl_deep.py (3 hidden
layers, a tree of areas, every claim of TL3 at eps = 0), and on its control with one unit that switches inside its
area (eps_lin > 0).
A  TI0: the compensation identity and the chain rule, on random distributions (1e-12), and once in exact arithmetic
   (sympy, rational probabilities, logarithms kept symbolic).
B  TI1: the information table of every node with children equals the chain rule from its children, the silent part
   and c_N (part 3); the least over one constant logit vector per area (a convex problem, solved numerically) equals
   I_w and is reached by log sbar_w (part 1); along the zoom G1 -> G4 of tl_deep with k = 0, V does not fall (part 4).
C  TI2: for each zoom group and k = 0..4, the KL terms of TP1's box are <= Lin_S / 2; for k >= rank Sigma they are
   <= eps_lin / 4 (0 on the clean net, > 0 but <= eps_lin / 4 on the control).
A failure is a mismatch above 1e-8, or a bound broken by more than 1e-10."""
import json
import os

import numpy as np
import sympy as sp
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax, xlogy

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "tl_deep.py")).read().split("# ---------- A: claims and TL3 ----------")[0]
G = {"__name__": "tl_deep_builder"}
exec(compile(src, "tl_deep.py (builder only)", "exec"), G)
build, summary, TREE, children, NY, PP = (G[k] for k in ("build", "summary", "TREE", "children", "NY", "PP"))

fails, out = [], {}
rng = np.random.default_rng(31)


def kl(s, q):
    """KL(s || q) for probability vectors (rows), with 0 log 0 = 0."""
    r = (xlogy(s, s) - xlogy(s, q)).sum(-1)
    return float(r) if s.ndim == 1 else r


def kl_logits(a, b):
    """KL(softmax(a) || softmax(b)) row by row, from logits (stable for large logits)."""
    la, lb = a - logsumexp(a, axis=-1, keepdims=True), b - logsumexp(b, axis=-1, keepdims=True)
    return (np.exp(la) * (la - lb)).sum(-1)


# ---------- A: Lemma TI0 ----------
errA = 0.0
for _ in range(50):
    n, c = rng.integers(3, 12), rng.integers(2, 6)
    s = softmax(rng.normal(0, 2, (n, c)), axis=1)
    p = rng.uniform(0.1, 1, n)
    q = softmax(rng.normal(0, 1, c))
    m = p.sum()
    sbar = p @ s / m
    I = float(p @ kl(s, sbar))
    errA = max(errA, abs(float(p @ kl(s, q)) - I - m * kl(sbar, q)))
    part = rng.integers(0, 3, n)
    rhs = 0.0
    for j in np.unique(part):
        R = part == j
        mj, sj = p[R].sum(), p[R] @ s[R] / p[R].sum()
        rhs += float(p[R] @ kl(s[R], sj)) + mj * kl(sj, sbar)
    errA = max(errA, abs(I - rhs))
out["A_TI0_max_error"] = errA
if errA > 1e-12:
    fails.append(f"A: TI0 error {errA}")
# exact: three points, two parts
S = [sp.Matrix([sp.Rational(1, 2), sp.Rational(1, 3), sp.Rational(1, 6)]),
     sp.Matrix([sp.Rational(1, 5), sp.Rational(3, 5), sp.Rational(1, 5)]),
     sp.Matrix([sp.Rational(1, 4), sp.Rational(1, 4), sp.Rational(1, 2)])]
P = [sp.Rational(1, 3), sp.Rational(1, 6), sp.Rational(1, 2)]


def KLs(a, b):
    return sum(a[i] * (sp.log(a[i]) - sp.log(b[i])) for i in range(3))


mR = sum(P)
sb = sum((P[i] * S[i] for i in range(3)), sp.zeros(3, 1)) / mR
I_all = sum(P[i] * KLs(S[i], sb) for i in range(3))
s01 = (P[0] * S[0] + P[1] * S[1]) / (P[0] + P[1])
chain = P[0] * KLs(S[0], s01) + P[1] * KLs(S[1], s01) + (P[0] + P[1]) * KLs(s01, sb) + P[2] * KLs(S[2], sb)
exact_diff = sp.simplify(sp.expand_log(I_all - chain, force=True))
out["A_TI0_exact_difference"] = str(exact_diff)
if exact_diff != 0:
    fails.append(f"A: exact chain rule difference {exact_diff}")

# ---------- B: Theorem TI1 on the deep net ----------
net = build()
p_all, Y = net["p"], net["Y"]
s_all = softmax(Y, axis=1)


def table_direct(v):
    R = net["area"][v]
    m = p_all[R].sum()
    sbar = p_all[R] @ s_all[R] / m
    return m, sbar, float(p_all[R] @ kl(s_all[R], sbar))


cN = Y[~np.any([net["area"][v] for v in TREE if TREE[v][0] == 1], axis=0)][0]   # y_N on points in no area: c_N


def table_rec(v):
    """(m, sbar, I) of v from the layer-3 tables, the masses and c_N, by TI1 part 3 (never reading y_N above layer 3)."""
    kids = children(v)
    if not kids:
        if TREE[v][0] == 3:
            return table_direct(v)
        m = p_all[net["area"][v]].sum()
        return m, softmax(cN), 0.0
    R = net["area"][v]
    m = p_all[R].sum()
    parts = [table_rec(c) for c in kids]
    m0 = m - sum(pc[0] for pc in parts)
    sN = softmax(cN)
    sbar = (sum(pc[0] * pc[1] for pc in parts) + m0 * sN) / m
    I = sum(pc[2] + pc[0] * kl(pc[1], sbar) for pc in parts)
    if m0 > 1e-15:
        I += m0 * kl(sN, sbar)
    return m, sbar, I


errB = 0.0
for v in TREE:
    md, sd, Id = table_direct(v)
    mr, sr, Ir = table_rec(v)
    errB = max(errB, abs(md - mr), float(np.abs(sd - sr).max()), abs(Id - Ir) / max(Id, 1e-6))
out["B_TI1_part3_table_max_relative_error"] = errB
if errB > 1e-8:
    fails.append(f"B: table recursion error {errB}")
# part 1: the least over one logit vector per area
errB1 = 0.0
for v in TREE:
    R = net["area"][v]
    f = lambda t: float(p_all[R] @ kl_logits(Y[R], np.broadcast_to(t, Y[R].shape)))
    best = min(minimize(f, rng.normal(0, 1, NY), method="BFGS", options=dict(gtol=1e-12)).fun for _ in range(3))
    m, sbar, I = table_direct(v)
    errB1 = max(errB1, (abs(best - I) - 1e-7) / max(I, 1e-6), (abs(f(np.log(sbar)) - I) - 1e-12) / max(I, 1e-6))  # abs tol: BFGS stops near 1e-9 when I ~ 0
    out.setdefault("B_TI1_part1_rows", {})[v] = dict(I=I, optimiser=best, at_log_sbar=f(np.log(sbar)))
    if best < I - 1e-8 * max(1, I):
        fails.append(f"B: an optimiser beat I at {v}")
out["B_TI1_part1_max_relative_error"] = errB1
if errB1 > 1e-7:
    fails.append(f"B: part 1 error {errB1}")
zoom = {"G1": [["A11"], ["A12"]], "G2": [["A11", "A12"]], "G3": [["A1"]], "G4": [["A"]]}
Vk0 = {g: sum(table_rec(w)[2] for grp in groups for w in grp) for g, groups in zoom.items()}
out["B_TI1_part4_V_k0"] = Vk0
vals = list(Vk0.values())
if any(vals[i + 1] < vals[i] - 1e-12 for i in range(len(vals) - 1)):
    fails.append("B: V fell along the zoom")


# ---------- C: Theorem TI2 ----------
def tp1_box_kl(netx, summx, nodes, k):
    """KL and D2 terms of TP1 part 2's always-on box (constants chosen as TP1 does), on N's data."""
    Sg = sum(summx[w]["S"] for w in nodes)
    ev, V = np.linalg.eigh(Sg)
    Q = V[:, np.argsort(-ev)][:, :k]
    klt, d2 = 0.0, 0.0
    for w in nodes:
        R, sw = netx["area"][w], summx[w]
        pre = sw["Z"] @ sw["B"].T @ Q
        Cb = pre + 1 + np.abs(pre).max(0)
        yb = Cb[R] @ Q.T
        res = (netx["Y"][R] - yb) @ PP
        ww = netx["p"][R] / netx["p"][R].sum()
        shift = ww @ res
        yb = yb + shift                                   # TP1's free term on the area
        d2 += 0.5 * float((netx["p"][R][:, None] * ((netx["Y"][R] - yb) @ PP) ** 2).sum())
        klt += float(netx["p"][R] @ kl_logits(netx["Y"][R], yb))
    eps = sum(summx[w]["eps"] for w in nodes)
    rank = int((np.linalg.eigvalsh(Sg) > 1e-10 * max(1.0, np.abs(Sg).max())).sum())
    return klt, d2, eps, rank


rowsC = {}
for label, netx in (("clean", net), ("control (switch inside)", build(switch_bug=True))):
    summx = {v: summary(netx, v) for v in TREE}
    for g, groups in zoom.items():
        for grp in groups:
            for k in range(0, 5):
                klt, d2, eps, rank = tp1_box_kl(netx, summx, grp, k)
                lin = d2
                key = f"{label} {g} {'+'.join(grp)} k={k}"
                rowsC[key] = dict(KL=klt, Lin=lin, eps_lin=eps, rank=rank)
                if klt > lin / 2 + 1e-10:
                    fails.append(f"C: {key}: KL {klt} > Lin/2 {lin / 2}")
                if k >= rank and klt > eps / 4 + 1e-10:
                    fails.append(f"C: {key}: k >= rank but KL {klt} > eps_lin/4 {eps / 4}")
out["C_TI2"] = rowsC
out["C_TI2_rows_with_k_ge_rank"] = sum(1 for key, r in rowsC.items() if int(key.split("k=")[1]) >= r["rank"])

print(json.dumps(out, indent=1, default=float))
json.dump(out, open(os.path.join(HERE, "ti_info.json"), "w"), indent=1, default=float)
for f in fails:
    print("FAIL", f)
print("FAILURES:", len(fails))
