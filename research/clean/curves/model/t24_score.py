# serves: T24
"""Exact check of Theorem T24 (COARSE-THEORY.md B.C): the score on the training law is not fixed by the network.

Network: T20's teacher, gap T(x) = relu(x - 1) + relu(-x - 1), points -3..3.
Part 1 (exact, sympy): every width-1 gap equal to T on {0, 1, 2, 3} is relu(x - 1). Units relu(s (x - t)) with the
kink at a point or in an open piece, both signs; constant builds separately.
Part 2 (exact, sympy, D2): for mu_3 ~ (1,1,3,3,3,1,1) and mu_4 ~ (3,1,1,1,1,1,3), the exact minimum of the weighted
least-squares residual V(t) on every piece (both signs, and the constant build), and the set of minimisers.
Lemma T24a (numeric sanity): on mu_1, the cross-entropy of N on its own labels is below that of 200 random predictors.
Prediction (written before the run): part 1 finds only relu(x - 1); part 2 gives 46/11 at t = 14/11 (and its mirror)
for mu_3, and 6 on kinks in [2, 3) (and the mirror) for mu_4; the two sets of units are disjoint.
"""
import itertools
import json
import os

import numpy as np
import sympy as sp

XS = list(range(-3, 4))
T = [max(x - 1, 0) + max(-x - 1, 0) for x in XS]
out = {}

# ---------- Part 1 ----------
P2 = [0, 1, 2, 3]
T2 = [T[XS.index(x)] for x in P2]
p, q, g = sp.symbols("p q g")
bounds = [-sp.oo] + P2 + [sp.oo]
pieces = [("pt", x0) for x0 in P2] + [("iv", (lo, hi)) for lo, hi in zip(bounds[:-1], bounds[1:])]
fits = []
for (kind, val), s in itertools.product(pieces, (1, -1)):
    vec = []
    for x in P2:
        if kind == "pt":
            vec.append(p * (x - val) if s * (x - val) > 0 else 0)
        else:
            lo, hi = val
            active = (x >= hi) if s == 1 else (x <= lo)
            vec.append(p * x - q if active else 0)
    unk = [p, g] + ([q] if kind == "iv" else [])
    for sol in sp.linsolve([vec[i] + g - T2[i] for i in range(4)], unk):
        d = dict(zip(unk, sol))
        if any(e.free_symbols for e in sol):
            fits.append((kind, str(val), s, "family", str(sol)))
            continue
        if d[p] == 0:
            continue
        if kind == "pt":
            fits.append((kind, str(val), s, str(sol)))
        else:
            tv = d[q] / d[p]
            if val[0] < tv < val[1]:
                fits.append((kind, str(val), s, str(sol), str(tv)))
print("Part 1: exact width-1 fits on {0,1,2,3}:", fits)
part1_ok = len(fits) == 1 and fits[0][0] == "pt" and fits[0][1] == "1" and fits[0][2] == 1
out["part1_fits"] = fits
out["part1_ok"] = part1_ok

# ---------- Part 2 ----------
t, a = sp.symbols("t a", real=True)


def piece_minima(w):
    """Exact min of V over each piece for s = +1 (s = -1 mirrors, since w and T are symmetric), plus the constant."""
    res = []
    bnd = [-sp.oo] + XS + [sp.oo]
    for lo, hi in zip(bnd[:-1], bnd[1:]):
        u = [(x - t) if x >= hi else 0 for x in XS]
        S = sum(wi * (a * ui + g - Ti) ** 2 for wi, ui, Ti in zip(w, u, T))
        sol = sp.solve([sp.diff(S, a), sp.diff(S, g)], [a, g], dict=True)
        if not sol:  # unit is zero on all points: constant build
            continue
        V = sp.simplify(S.subs(sol[0]))
        cands = []
        if sp.simplify(sp.diff(V, t)) == 0:
            cands.append(("const-in-t", sp.nsimplify(V)))
        else:
            for c in sp.solve(sp.diff(V, t), t):
                if c.is_real and (lo == -sp.oo or c > lo) and (hi == sp.oo or c < hi):
                    cands.append((c, sp.nsimplify(V.subs(t, c))))
        for e in (lo, hi):
            if e not in (-sp.oo, sp.oo):
                cands.append((e, sp.nsimplify(sp.limit(V, t, e))))
        if lo == -sp.oo:
            cands.append(("-oo", sp.nsimplify(sp.limit(V, t, -sp.oo))))
        for c, v in cands:
            res.append({"piece": (str(lo), str(hi)), "t": str(c), "V": v})
    S0 = sum(wi * (g - Ti) ** 2 for wi, Ti in zip(w, T))
    g0 = sp.solve(sp.diff(S0, g), g)[0]
    res.append({"piece": "constant", "t": "-", "V": sp.nsimplify(S0.subs(g, g0))})
    return res


summary = {}
for name, w in (("mu3_middle", [1, 1, 3, 3, 3, 1, 1]), ("mu4_ends", [3, 1, 1, 1, 1, 1, 3])):
    w = [sp.Rational(v) for v in w]
    res = piece_minima(w)
    best = min(r["V"] for r in res)
    argmin = [r for r in res if r["V"] == best]
    second = min(r["V"] for r in res if r["V"] != best)
    print(f"Part 2 {name}: least residual {best} (= {float(best):.6f}); next value {second} (= {float(second):.6f})")
    for r in argmin:
        print("     reached at", r)
    summary[name] = {"best": str(best), "second": str(second), "argmin": [{k: str(v) for k, v in r.items()} for r in argmin]}
mu3 = summary["mu3_middle"]
mu4 = summary["mu4_ends"]
mu3_ok = mu3["best"] == "46/11" and all(r["t"] == "14/11" for r in mu3["argmin"])
# mu4: every minimiser has its kink in [2, 3) (data vector a multiple of the indicator of x = 3) -- the piece (2, 3)
# constant in t, or its end t = 2; the end t = 3 is a limit where the unit vanishes (not a build of value 6)
# (the end t = 2 of the piece (1, 2) is the same unit as a kink exactly at 2: data vector = indicator of x = 3)
mu4_ok = mu4["best"] == "6" and all(
    (r["piece"] == "('2', '3')" and r["t"] in ("const-in-t", "2", "3")) or (r["piece"] == "('1', '2')" and r["t"] == "2")
    for r in mu4["argmin"])
part2_ok = mu3_ok and mu4_ok
print("Part 2: mu3 minimiser t = 14/11 only:", mu3_ok, "; mu4 minimisers have kink in [2, 3):", mu4_ok)
out["part2"] = summary
out["part2_ok"] = part2_ok

# ---------- Lemma T24a sanity ----------
rng = np.random.default_rng(0)
qN = 1 / (1 + np.exp(-np.array(T, float)))


def ce(pv):
    return float(np.mean(-(qN * np.log(pv) + (1 - qN) * np.log(1 - pv))))


ceN = ce(qN)
worse = all(ce(np.clip(rng.uniform(0.01, 0.99, 7), 1e-6, 1 - 1e-6)) >= ceN for _ in range(200))
print("Lemma T24a: cross-entropy of N on its own labels below 200 random predictors:", worse)
out["lemma_ok"] = worse

# ---------- Definition D20: the network is perfectly clean for the mirror ----------
W1 = np.array([[1.0], [-1.0]]); b1 = np.array([-1.0, -1.0])      # units relu(x - 1), relu(-x - 1)
Wout = np.array([[1.0, 1.0], [0.0, 0.0]]); c = np.zeros(2)          # logits (T, 0)
rho0 = np.array([[-1.0]]); pi = np.array([[0.0, 1.0], [1.0, 0.0]]); sigma = np.eye(2)
eps = max(np.linalg.norm(W1 @ rho0 - pi @ W1) / np.linalg.norm(W1),
          np.linalg.norm(pi @ b1 - b1) / (np.linalg.norm(b1) + np.linalg.norm(W1)),
          np.linalg.norm(Wout @ pi - sigma @ Wout) / np.linalg.norm(Wout))
read_ok = np.all(np.abs(Wout).sum(0) > 0)
d20_ok = eps == 0.0 and read_ok
print("Definition D20: eps_mirror(N) =", eps, "; every unit read:", read_ok)
out["d20_ok"] = bool(d20_ok)

ALL = part1_ok and part2_ok and worse and d20_ok
out["all_ok"] = ALL
print("ALL OK:", ALL)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "t24_score.json"), "w"), indent=1, default=str)
