# serves: T26
"""Exact check of Theorem T26 part 2 (COARSE-THEORY.md B.Z): folds use the weights.

N_1: units relu(x - 1), relu(-x - 1), gap T. N_2: units relu(x + 3), relu(x + 1), relu(x - 1), output weights -1, 1, 1,
offset 2. Mix P': points -3..3, weights (3,3,3,1,1,1,1). D2, width 1.
Folds of N_1 = rising units with kink >= 1, falling units with kink <= -1 (and 0). Folds of N_2 = rising units with
kink >= -3 (and 0). Exact least weighted residual over each allowed set, piece by piece (sympy).
Prediction (written before the run): N_2's gap equals T on the points; least values 26/7 (N_1 folds; equal to all
width-1 builds) and 84/11 (N_2 folds).
"""
import json
import os

import sympy as sp

XS = list(range(-3, 4))
T = [max(x - 1, 0) + max(-x - 1, 0) for x in XS]
relu = lambda v: max(v, 0)
N2 = [2 - relu(x + 3) + relu(x + 1) + relu(x - 1) for x in XS]
same = N2 == T
print("N_2's gap on the points:", N2, "equals T:", same)
t, a, g = sp.symbols("t a g", real=True)


def clip(lo, hi, L, U):
    lo2 = lo if L == -sp.oo else (L if lo == -sp.oo else max(lo, L))
    hi2 = hi if U == sp.oo else (U if hi == sp.oo else min(hi, U))
    if lo2 != -sp.oo and hi2 != sp.oo and lo2 > hi2:
        return None
    return (lo2, hi2)


def least(w, allow):
    vals = []
    bnd = [-sp.oo] + XS + [sp.oo]
    for s in (1, -1):
        for lo, hi in zip(bnd[:-1], bnd[1:]):
            A = allow(s, lo, hi)
            if A is None:
                continue
            L, U = A
            u = [(s * (x - t)) if ((x >= hi) if s == 1 else (x <= lo)) else 0 for x in XS]
            S = sum(wi * (a * ui + g - Ti) ** 2 for wi, ui, Ti in zip(w, u, T))
            sol = sp.solve([sp.diff(S, a), sp.diff(S, g)], [a, g], dict=True)
            if not sol:
                continue
            V = sp.simplify(S.subs(sol[0]))
            if sp.simplify(sp.diff(V, t)) == 0:
                vals.append(sp.nsimplify(V))
            else:
                for r in sp.solve(sp.diff(V, t), t):
                    if r.is_real and (L == -sp.oo or r >= L) and (U == sp.oo or r <= U):
                        vals.append(sp.nsimplify(V.subs(t, r)))
            for e in (L, U):
                if e not in (-sp.oo, sp.oo):
                    vals.append(sp.nsimplify(sp.limit(V, t, e)))
    S0 = sum(wi * (g - Ti) ** 2 for wi, Ti in zip(w, T))
    vals.append(sp.nsimplify(S0.subs(g, sp.solve(sp.diff(S0, g), g)[0])))
    return min(vals, key=float)


w = [sp.Rational(v) for v in (3, 3, 3, 1, 1, 1, 1)]
v_any = least(w, lambda s, lo, hi: (lo, hi))
v_n1 = least(w, lambda s, lo, hi: clip(lo, hi, 1, sp.oo) if s == 1 else clip(lo, hi, -sp.oo, -1))
v_n2 = least(w, lambda s, lo, hi: clip(lo, hi, -3, sp.oo) if s == 1 else None)
print("least over all width-1 builds:", v_any, "; over N_1 folds:", v_n1, "; over N_2 folds:", v_n2)
ok = same and v_n1 == sp.Rational(26, 7) and v_any == v_n1 and v_n2 == sp.Rational(84, 11)
print("ALL OK:", ok)
json.dump({"same_on_P": same, "any": str(v_any), "N1": str(v_n1), "N2": str(v_n2), "all_ok": bool(ok)},
          open(os.path.join(os.path.dirname(__file__), "t26_weights.json"), "w"), indent=1)
