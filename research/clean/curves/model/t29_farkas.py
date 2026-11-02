# serves: T29
"""Check of Theorem T29 (COARSE-THEORY.md B.Z): a unit covers a group iff it is a fold (Farkas).

Random groups (s = 3 members in d = 4) with a non-empty common silent set Z_S. For random units (w, b):
membership in the fold cone {sum m_i (w_i, b_i) + tau (0, -1), m, tau >= 0} by a linear program, and covering by a
second linear program (max of w^T x + b over Z_S intersected with a large box). Fold => covers; not a fold => some
x in Z_S with w^T x + b > 0 (the box is large enough when the maximiser is finite; unbounded cases count as "fires").
Prediction (written before the run): the two tests agree on every unit.
"""
import json
import os

import numpy as np
from scipy.optimize import linprog

rng = np.random.default_rng(0)
agree = 0
total = 0
for trial in range(40):
    d, s = 4, 3
    Wg = rng.normal(size=(s, d)); bg = rng.normal(size=s)
    # non-empty Z_S: feasibility of W x + b <= 0
    if linprog(np.zeros(d), A_ub=Wg, b_ub=-bg, bounds=[(None, None)] * d, method="highs").status != 0:
        continue
    for k in range(10):
        if k < 5:  # a fold
            m = rng.uniform(0, 2, s); tau = rng.uniform(0, 1)
            w = m @ Wg; b = m @ bg - tau
        else:      # a random unit
            w = rng.normal(size=d); b = rng.normal()
        # fold membership: exists m >= 0, tau >= 0 with sum m_i w_i = w, sum m_i b_i - tau = b
        Aeq = np.vstack([np.hstack([Wg.T, np.zeros((d, 1))]), np.hstack([bg, [-1.0]])])
        beq = np.concatenate([w, [b]])
        is_fold = linprog(np.zeros(s + 1), A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * (s + 1), method="highs").status == 0
        # covers: max w^T x + b over Z_S (box 1e6) <= 1e-9
        r = linprog(-w, A_ub=Wg, b_ub=-bg, bounds=[(-1e6, 1e6)] * d, method="highs")
        covers = (r.status == 0) and (-r.fun + b <= 1e-7)
        total += 1
        agree += int(is_fold == covers)
print(f"fold test and covering test agree on {agree} of {total} units")
ok = agree == total and total > 0
print("ALL OK:", ok)
json.dump({"agree": agree, "total": total, "all_ok": bool(ok)},
          open(os.path.join(os.path.dirname(__file__), "t29_farkas.json"), "w"), indent=1)
