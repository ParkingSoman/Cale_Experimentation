# serves: TA6 (UNIT-BRANCH.md)
"""Exact instance of Proposition TA6: a unit that no reader reads still breaks U4 with a set it fires with.
Two logits, input x in R, P = {1, 2} uniform. u = relu(x) with gap weight 1 (N's gap is relu(x)).
v = relu(x - 1/2) with column (1, 1): it writes the same to both logits, so it has edges (DU1) but no
class probability depends on it (softmax-unread). S = {u}, T = {v}.
Box for T at budget 1: one unit relu(e z_v), e >= 0 (a fold: covering, output-faithful, silence-preserving),
with a new write c to the gap. D2 exact with fractions; D_P (KL) by a 1-d search."""
import json
import math
import os
from fractions import Fraction as Fr

rl = lambda t: t if t > 0 else 0
P = [Fr(1), Fr(2)]
p = [Fr(1, 2), Fr(1, 2)]
hu = [rl(x) for x in P]
hv = [rl(x - Fr(1, 2)) for x in P]
gN = hu[:]                                     # v's column (1, 1) adds nothing to the gap

D2 = lambda gb: sum(pi * (a - b) ** 2 for pi, a, b in zip(p, gN, gb))
D2_S0 = D2([0, 0])                             # S removed
D2_T1 = D2(gN)                                 # v kept with its own write 0: exact
D2_T0 = D2(gN)                                 # v removed: the gap is unchanged (softmax too)
D2_rem_both = D2([0, 0])                       # S and v removed
A = sum(pi * (a - 0) * h for pi, a, h in zip(p, gN, hv))
B = sum(pi * h * h for pi, h in zip(p, hv))
c_best = A / B
D2_joint = D2([c_best * h for h in hv])        # S removed, v's box writes c_best * h_v
out = dict(hu=[str(t) for t in hu], hv=[str(t) for t in hv],
           D2_S0=str(D2_S0), D2_T0=str(D2_T0), D2_T1=str(D2_T1),
           I_D2_budget0=str(D2_rem_both - D2_S0 - D2_T0),
           c_best=str(c_best), D2_joint=str(D2_joint), I_D2_budget1=str(D2_joint - D2_S0 - D2_T1))


def kl_gap(t, b):          # two-logit KL(softmax(gap t) || softmax(gap b))
    q = 1 / (1 + math.exp(-t)); r = 1 / (1 + math.exp(-b))
    return q * math.log(q / r) + (1 - q) * math.log((1 - q) / (1 - r))


DP = lambda gb: sum(float(pi) * kl_gap(float(a), float(b)) for pi, a, b in zip(p, gN, gb))
lo, hi = -10.0, 10.0
for _ in range(200):
    m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
    f = lambda c: DP([c * float(h) for h in hv])
    lo, hi = (lo, m2) if f(m1) < f(m2) else (m1, hi)
cK = (lo + hi) / 2
DP_S0 = DP([0, 0]); DP_joint = DP([cK * float(h) for h in hv])
# TU7's gamma_v: sum_x p_x h_v(x) g_x, g_x = softmax change when S is removed (first logit coordinate)
gam = sum(float(pi) * float(h) * (0.5 - 1 / (1 + math.exp(-float(a)))) for pi, h, a in zip(p, hv, gN))
out.update(DP_S0=DP_S0, DP_T1=0.0, c_KL=cK, DP_joint=DP_joint, I_DP_budget1=DP_joint - DP_S0, gamma_v_first=gam)
out["prediction"] = "I_D2_budget0 = 0; I_D2_budget1 = -49/20 < 0; I_DP_budget1 < 0; gamma_v != 0"
out["held"] = (D2_rem_both - D2_S0 - D2_T0 == 0 and D2_joint - D2_S0 - D2_T1 == Fr(-49, 20)
               and DP_joint - DP_S0 < 0 and gam != 0)
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ta6_unread.json"), "w"), indent=1)
