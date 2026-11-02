# serves: T38.2
"""Exact check of the orthogonal split in Corollary T38.2 (COARSE-THEORY.md B.O), with sympy.

Symbols: r = 3 readers, family output a, write c, bias shift d, unit output u (a scalar). With
t = c^T a / |c|^2, gamma = c^T d / |c|^2, a' = a - t c, d' = d - gamma c, check the identity
|a - c u - d|^2 = |c|^2 (t - u - gamma)^2 + |a' - d'|^2, and that the second term has no u.
Prediction (written before the run): the identity holds exactly.
"""
import json
import os

import sympy as sp

a = sp.Matrix(sp.symbols("a0:3")); c = sp.Matrix(sp.symbols("c0:3")); d = sp.Matrix(sp.symbols("d0:3"))
u = sp.Symbol("u")
cc = (c.T * c)[0]
t = (c.T * a)[0] / cc; g = (c.T * d)[0] / cc
ap = a - t * c; dp = d - g * c
lhs = ((a - c * u - d).T * (a - c * u - d))[0]
rhs = cc * (t - u - g) ** 2 + ((ap - dp).T * (ap - dp))[0]
ok = sp.simplify(lhs - rhs) == 0 and u not in ((ap - dp).T * (ap - dp))[0].free_symbols
print("orthogonal split holds exactly:", ok)
print("ALL OK:", ok)
json.dump({"all_ok": bool(ok)}, open(os.path.join(os.path.dirname(__file__), "t38_write.json"), "w"))
