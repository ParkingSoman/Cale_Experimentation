# serves: TI3
# serves: TI4
# serves: TI5
"""Theorems TI3, TI4 and Proposition TI5 (COARSE-THEORY.md B.I) on the exact instances of TI5.
Instances of TI5 part 2: four points (+-1, 0), (0, +-1) with weights 1/4, two always-on units z = x + 2, three logits
y = a F (z - 2) + y0 with F = (f1, f2), y0 = 0 (N_a, uniform mean softmax) or log(1/2, 1/4, 1/4) (N'_a).
For k = 1 the least over boxes on throughout (TI3 part 2) is a least over lines q in the plane of centred vectors of a
convex problem in (read alpha, constant t); the line is searched on a grid of 360 angles plus a local refinement.
A  TI3 part 1: at the computed least, the build's mean softmax equals N's (to 1e-6).
B  TI4: L_on(a) / a^2 -> Phi = 1/12 (N_a) and 1/16 (N'_a), with (L_on / a^2 - Phi) / a bounded; in N'_a the least line
   tends to (2, -1, -1); the D2 build along (0, 1, -1) has KL / a^2 -> 3/32; in N_a the spread over lines -> 0.
C  TI5 part 1: the closed forms and their order (exact integers for 1a, concavity for 1b), and the least over one
   constant logit vector equals the information I (TI1 part 1) for each of the four networks.
D  (reported only; the open question of the Comment of TI3) one unit that may switch, relu(m^T z + beta), with free
   logit weights and bias, found by a local optimiser with restarts, against L_on, at a = 0.5 and a = 1.
A failure is a mismatch beyond the stated tolerances."""
import json
import os

import numpy as np
import sympy as sp
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp, softmax

HERE = os.path.dirname(os.path.abspath(__file__))
fails, out = [], {}
rng = np.random.default_rng(2031)

X = np.array([[1., 0], [-1, 0], [0, 1], [0, -1]])
p = np.full(4, 0.25)
f1 = np.array([1., -1, 0]) / np.sqrt(2)
f2 = np.array([1., 1, -2]) / np.sqrt(6)
F = np.stack([f1, f2], 1)
Y0 = {"N_a (uniform)": np.zeros(3), "N'_a (1/2,1/4,1/4)": np.log([0.5, 0.25, 0.25])}
PHI = {"N_a (uniform)": 1 / 12, "N'_a (1/2,1/4,1/4)": 1 / 16}


def kl_logits(a, b):
    la, lb = a - logsumexp(a, axis=-1, keepdims=True), b - logsumexp(b, axis=-1, keepdims=True)
    return (np.exp(la) * (la - lb)).sum(-1)


def logits_N(a, y0):
    return a * X @ F.T + y0


def inner(q, yN, x0=None):
    """least over alpha (2) and t (3) of sum p KL(softmax(yN) || softmax(t + q alpha^T x)): convex."""
    def f(v):
        b = v[2:][None, :] + np.outer(X @ v[:2], q)
        return float(p @ kl_logits(yN, b))
    best = None
    for s in range(2):
        v0 = x0 if (x0 is not None and s == 0) else rng.normal(0, 0.1, 5)
        r = minimize(f, v0, method="BFGS", options=dict(gtol=1e-13, maxiter=2000))
        if best is None or r.fun < best.fun:
            best = r
    return best


def line(theta):
    return F @ np.array([np.cos(theta), np.sin(theta)])


def L_on(a, y0):
    yN = logits_N(a, y0)
    grid = np.linspace(0, np.pi, 360, endpoint=False)
    vals = np.array([inner(line(th), yN).fun for th in grid])
    i = int(np.argmin(vals))
    r = minimize_scalar(lambda th: inner(line(th), yN).fun, bracket=(grid[i] - 0.01, grid[i], grid[i] + 0.01),
                        tol=1e-10)
    th = r.x if r.fun <= vals[i] else grid[i]
    best = inner(line(th), yN)
    return best.fun, th, best, vals


resB, resA = {}, {}
for name, y0 in Y0.items():
    rows = {}
    for a in (0.4, 0.2, 0.1, 0.05):
        val, th, best, vals = L_on(a, y0)
        q = line(th)
        q3 = q / np.linalg.norm(q)
        target = np.array([2., -1, -1]) / np.sqrt(6)
        rows[a] = dict(L_over_a2=val / a ** 2, gap_over_a=(val / a ** 2 - PHI[name]) / a,
                       line_cos_to_2m1m1=float(abs(q3 @ target)), spread_over_lines_a2=float(np.ptp(vals) / a ** 2))
        # A: means match at the least
        yN = logits_N(a, y0)
        b = best.x[2:][None, :] + np.outer(X @ best.x[:2], q)
        mean_gap = float(np.abs(p @ softmax(b, axis=1) - p @ softmax(yN, axis=1)).max())
        resA[f"{name} a={a}"] = mean_gap
        if mean_gap > 1e-6:
            fails.append(f"A: {name} a={a}: mean softmax mismatch {mean_gap}")
        # D2 build along (0, 1, -1) (TP1 box: centred logits ybar + q q^T Ytil)
        if name.startswith("N'"):
            qd = np.array([0., 1, -1]) / np.sqrt(2)
            Yt = a * X @ F.T
            bD2 = y0 + Yt @ np.outer(qd, qd)
            rows[a]["D2_line_KL_over_a2"] = float(p @ kl_logits(yN, bD2)) / a ** 2
    resB[name] = rows
    small = rows[0.05]
    if abs(small["L_over_a2"] - PHI[name]) > 0.05 * PHI[name]:
        fails.append(f"B: {name}: L/a^2 at a=0.05 is {small['L_over_a2']}, Phi {PHI[name]}")
    if max(abs(r["gap_over_a"]) for r in rows.values()) > 1.0:
        fails.append(f"B: {name}: (L/a^2 - Phi)/a not bounded by 1")
    if name.startswith("N'"):
        if small["line_cos_to_2m1m1"] < 0.99:
            fails.append(f"B: least line not near (2,-1,-1): cos {small['line_cos_to_2m1m1']}")
        if abs(small["D2_line_KL_over_a2"] - 3 / 32) > 0.05 * 3 / 32:
            fails.append(f"B: D2 line KL/a^2 {small['D2_line_KL_over_a2']} vs 3/32")
    elif small["spread_over_lines_a2"] > 0.05 * PHI[name]:
        fails.append(f"B: uniform case: spread over lines {small['spread_over_lines_a2']}")
out["A_TI3_means_match_max_gap"] = resA
out["B_TI4"] = resB

# ---------- C: TI5 part 1 ----------
l3, l2, l5, l7 = sp.log(3), sp.log(2), sp.log(5), sp.log(7)
Rr = sp.Rational
LN = Rr(1, 2) * (Rr(9, 10) * sp.log(Rr(9, 10) / Rr(7, 10)) + Rr(1, 10) * sp.log(Rr(1, 10) / Rr(3, 10))
                 + Rr(1, 2) * sp.log(Rr(1, 2) / Rr(7, 10)) + Rr(1, 2) * sp.log(Rr(1, 2) / Rr(3, 10)))
LNp = Rr(3, 4) * l3 - l2
expanded = sp.simplify(sp.expand_log(LN - (Rr(3, 5) * l3 + Rr(1, 2) * l5 - Rr(7, 10) * l7), force=True))
out["C_1a"] = dict(L_N=float(LN), L_Nprime=float(LNp), expansion_check=str(expanded),
                   integer_order=bool(3 ** 3 * 7 ** 14 > 2 ** 20 * 5 ** 10))
if expanded != 0 or not (float(LNp) > float(LN)):
    fails.append("C: 1a closed forms")
phi = lambda t: float(np.log(2) + t / (1 + np.exp(-t)) - np.log1p(np.exp(t)))
out["C_1b"] = dict(L_N=phi(1.0), L_Nprime=0.5 * phi(np.sqrt(2)))
if not phi(1.0) > 0.5 * phi(np.sqrt(2)):
    fails.append("C: 1b order")


def least_const(yN, w):
    f = lambda t: float(w @ kl_logits(yN, np.broadcast_to(t, yN.shape)))
    return min(minimize(f, rng.normal(0, 1, 2), method="BFGS", options=dict(gtol=1e-12)).fun for _ in range(3))


h = np.array([2 * np.log(3), 0.0])
cases = {"1a N": (np.stack([h, 0 * h], 1), np.full(2, .5), float(LN)),
         "1a N'": (np.stack([h - np.log(3), 0 * h], 1), np.full(2, .5), float(LNp)),
         "1b N": (np.stack([np.array([-1., 1]), np.zeros(2)], 1), np.full(2, .5), phi(1.0)),
         "1b N'": (np.stack([np.array([-np.sqrt(2), 0, np.sqrt(2)]), np.zeros(3)], 1), np.array([.25, .5, .25]),
                   0.5 * phi(np.sqrt(2)))}
rowsC = {}
for k, (yN, w, val) in cases.items():
    got = least_const(yN, w)
    rowsC[k] = dict(closed_form=val, optimiser=got)
    if abs(got - val) > 1e-9:
        fails.append(f"C: {k}: least {got} vs closed form {val}")
out["C_least_constant"] = rowsC


# ---------- D: a box that may switch (reported only) ----------
def switch_box(a, y0, restarts=40):
    yN = logits_N(a, y0)
    Z = X + 2

    def f(v):
        u = np.maximum(Z @ v[:2] + v[2], 0)
        b = v[6:][None, :] + np.outer(u, v[3:6])
        return float(p @ kl_logits(yN, b))
    best = np.inf
    for _ in range(restarts):
        r = minimize(f, rng.normal(0, 1.5, 9), method="BFGS", options=dict(gtol=1e-12, maxiter=4000))
        best = min(best, r.fun)
    return best


rowsD = {}
for name, y0 in Y0.items():
    for a in (0.5, 1.0):
        lon = L_on(a, y0)[0]
        sw = switch_box(a, y0)
        rowsD[f"{name} a={a}"] = dict(L_on=lon, best_switching_box_found=sw, ratio=sw / lon)
out["D_switching_box_reported_only"] = rowsD

print(json.dumps(out, indent=1, default=float))
json.dump(out, open(os.path.join(HERE, "ti_open.json"), "w"), indent=1, default=float)
for f_ in fails:
    print("FAIL", f_)
print("FAILURES:", len(fails))
