"""Step 7: choose a scope by code from a goal, before any reading. Then check the firing limit: remove weak firings of the scope's nodes, and measure KL on the goal.
Usage: scope_v1.py NAME PER_LAYER CLASSES... where CLASSES are ranges like 151-268. Reading photos only. Writes checks/scope_NAME.json."""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from mapper import scope  # noqa: E402
from networks import get  # noqa: E402

name, per = sys.argv[1], int(sys.argv[2])
classes = []
for part in sys.argv[3:]:
    a, _, b = part.partition("-")
    classes += list(range(int(a), int(b or a) + 1))
net = get("resnet18")
goal = dict(classes=classes)
rng = random.Random(7)
control = sorted(rng.sample([c for c in range(1000) if c not in set(classes)], len(classes)))
ex = net.goal_examples(goal, "read")
sample = np.sort(np.random.default_rng(0).choice(ex, min(1500, len(ex)), replace=False))
nodes, info = scope.trace(net, sample, per, log=lambda m: print(m, flush=True))
small = sample[:: max(1, len(sample) // 100)]
limit = net.weak_firing_by_layer(nodes, (.1, .25, .5), small, "read")
print("KL on the goal, removing firings under a share of each scope node's largest, one layer at a time:", flush=True)
for layer, row in limit.items():
    print(layer, row, flush=True)
out = dict(nodes=nodes, goal=dict(goal, name=name), control_goal=dict(classes=control), threshold=dict(per_layer=per, method="exact inputs from the outputs, then closure on the strongest input"), firings="strong firings and both poles, plus any weaker range the goal depends on", info=info, firing_limit_kl=limit)
with open(os.path.join(ROOT, "checks", f"scope_{name}.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(dict(count=len(nodes), info=info, firing_limit_kl=limit), indent=1))
