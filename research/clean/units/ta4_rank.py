# serves: TA4 (UNIT-BRANCH.md)
"""Check of Lemma TA4: the removal cost of a last-layer unit is set by its out-edge (activation x write), seen
through the KL curvature, not by its activation. Two logits, one input with N's gap 0 (offset chosen).
u: activation 2, gap weight 1/10.  v: activation 1, gap weight 1.  z: activation 5, write (1, 1) (along the
all-ones direction, which no logit difference sees)."""
import json
import math
import os


def kl_gap(t, b):
    q = 1 / (1 + math.exp(-t)); r = 1 / (1 + math.exp(-b))
    return q * math.log(q / r) + (1 - q) * math.log((1 - q) / (1 - r))


units = dict(u=(2, (0.1, 0)), v=(1, (1, 0)), z=(5, (1, 1)))
out = {}
for k, (h, w) in units.items():
    gap_w = w[0] - w[1]
    out[k] = dict(activation=h, D2_cost=(h * gap_w) ** 2, KL_cost=kl_gap(0, -h * gap_w),
                  second_order=0.5 * 0.25 * (h * gap_w) ** 2)
order_act = sorted(units, key=lambda k: -units[k][0])
order_cost = sorted(units, key=lambda k: -out[k]["KL_cost"])
out["order_by_activation"] = order_act
out["order_by_cost"] = order_cost
out["prediction"] = "activation order z, u, v; cost order v, u, z; z costs exactly 0"
out["held"] = order_act == ["z", "u", "v"] and order_cost == ["v", "u", "z"] and out["z"]["KL_cost"] == 0
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ta4_rank.json"), "w"), indent=1)
