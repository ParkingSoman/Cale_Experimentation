# serves: TU10 (UNIT-BRANCH.md)
"""Exact instance of Theorem TU10: gap |x| = relu(x) + relu(-x), P = {-2, -1, 1, 2} uniform, D2.
S = {relu(x)}, T = {relu(-x)}. Free: joint exact with the spare unit relu(-z_S); covering spare relu(m z_S + b),
m >= 0, b <= 0, is 0 on T's support, so it cannot help."""
import json
import os
from fractions import Fraction as Fr

rl = lambda t: t if t > 0 else Fr(0)
P = [Fr(-2), Fr(-1), Fr(1), Fr(2)]
g = [rl(x) + rl(-x) for x in P]
D = lambda gb: sum((a - b) ** 2 for a, b in zip(g, gb)) / len(P)
D_T0 = D([rl(x) for x in P])                       # T removed, S kept
D_S2 = D(g)                                        # S copied (plus an unused spare): exact
D_joint = D([rl(x) + rl(-x) for x in P])           # copy of relu(x) + spare relu(-z_S) = relu(-x), T removed
cover_spare_on_T = [rl(1 * x + 0) for x in P if x < 0]  # any m >= 0, b <= 0: here m = 1, b = 0
out = dict(D_T0=str(D_T0), D_S2=str(D_S2), D_joint=str(D_joint), I_free=str(D_joint - D_T0 - D_S2),
           covering_spare_values_on_T_support=[str(v) for v in cover_spare_on_T])
print(out)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "u_tu10.json"), "w"), indent=1)
