"""The firing limit, one layer at a time: remove firings under a share of each scope node's largest, on that layer's scope nodes only, and measure KL on the goal. Reading photos only. Usage: firing_limit.py NAME"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from networks import get  # noqa: E402

path = os.path.join(ROOT, "checks", f"scope_{sys.argv[1]}.json")
scope = json.load(open(path))
net = get("resnet18")
ex = net.goal_examples(scope["goal"], "read")
small = np.sort(np.random.default_rng(0).choice(ex, 96, replace=False))
limit = net.weak_firing_by_layer(scope["nodes"], (.1, .25, .5), small, "read")
for layer, row in limit.items():
    print(layer, row, flush=True)
scope["firing_limit_kl_by_layer"] = limit
scope.pop("firing_limit_kl", None)
json.dump(scope, open(path, "w"), indent=1)
