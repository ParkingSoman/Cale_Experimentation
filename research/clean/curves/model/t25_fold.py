# serves: T25
"""Check of Theorem T25 (COARSE-THEORY.md B.Z): a least build need not be a fold.

d = 2. N: units relu(x1 - 1), relu(-x1 - 1), logits (T(x1), 0). P: uniform on (s, T(s)), s = -3..3. Gamma: x1 -> -x1.
1. Definition D20 for N and the mirror: the equivariance defects are exactly 0, pi is not the identity, every unit read;
   P is invariant (exact integers).
2. The width-1 build with gap relu(x2) equals T on P (exact).
3. Fold builds depend on x1 only: the least D_P and D2_P over width-1 fold builds equals the least over width-1
   builds of T20's one-dimensional instance; computed with the exact rule of T21 (trunk_exact.exact_min, solver
   accuracy) and compared with T20's certified lower bound for the units of T20 part 1.
Prediction (written before the run): 1 and 2 hold exactly; 3 gives a positive least value (about the T20 value), so no
least build of width 1 is a fold.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from trunk_exact import exact_min  # noqa: E402

T = lambda s: max(s - 1, 0) + max(-s - 1, 0)
S = list(range(-3, 4))
P = [(s, T(s)) for s in S]
out = {}

# 1. D20
W1 = np.array([[1, 0], [-1, 0]]); b1 = np.array([-1, -1])
rho0 = np.diag([-1, 1]); pi = np.array([[0, 1], [1, 0]])
Wout = np.array([[1, 1], [0, 0]])
d20 = (np.array_equal(W1 @ rho0, pi @ W1) and np.array_equal(pi @ b1, b1) and np.array_equal(Wout @ pi, Wout)
       and not np.array_equal(pi, np.eye(2)) and np.all(np.abs(Wout).sum(0) > 0))
sym = sorted((-a, b) for a, b in P) == sorted(P)
print("1. D20 for the mirror (exact):", d20, "; P mirror-invariant:", sym)

# 2. relu(x2) on P
fit = all(max(b, 0) == T(a) for a, b in P)
print("2. gap relu(x2) equals T on P:", fit)

# 3. least fold builds (fold units read x1 only)
X = np.array([[float(a)] for a, _ in P]); pi_w = np.full(7, 1 / 7); t = np.array([float(T(a)) for a, _ in P])
vals = {}
for kind in ("kl", "d2"):
    (v, z, ch), _ = exact_min(X, pi_w, t, 1, kind, starts=3)
    vals[kind] = v
    print(f"3. least width-1 fold build, {kind}: {v:.6f}")
pos = vals["kl"] > 1e-3 and vals["d2"] > 1e-3
out.update({"d20": bool(d20), "P_symmetric": bool(sym), "relu_x2_fits": bool(fit), "least_fold": vals,
            "fold_positive": bool(pos)})
ALL = d20 and sym and fit and pos
out["all_ok"] = bool(ALL)
print("ALL OK:", ALL)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "t25_fold.json"), "w"), indent=1)
