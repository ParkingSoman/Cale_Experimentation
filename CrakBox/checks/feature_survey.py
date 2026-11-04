"""What the sheet's features look like across the network, on a random sample per layer. Workflow triggers are read from this and not guessed. Reading photos only."""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from mapper import sheets  # noqa: E402
from networks import get  # noqa: E402

net = get("resnet18")
rng = random.Random(0)
rows = []
PER = int(sys.argv[1]) if len(sys.argv) > 1 else 12
for layer in net.layers():
    for node in rng.sample(net.nodes(layer), PER):
        rows.append(sheets.features(net, node))
    print(layer, len(rows), flush=True)
with open(os.path.join(ROOT, "checks", "feature_survey.json"), "w") as f:
    json.dump(rows, f)
keys = ["largest", "strength", "on_share", "cliff", "output_effect", "neg_ratio", "top_input_share", "inputs_for_half", "route_split", "position_share", "label_spread", "reader_mix"]
print("all layers: quartiles")
for k in keys:
    a = np.array([r[k] for r in rows], float)
    print(f"{k:16s} {np.percentile(a, 10):8.3f} {np.percentile(a, 25):8.3f} {np.percentile(a, 50):8.3f} {np.percentile(a, 75):8.3f} {np.percentile(a, 90):8.3f}")
print("medians by layer:", " ".join(keys[:6]))
for layer in net.layers():
    v = [r for r in rows if net.layer_of(r["node"]) == layer]
    print(layer, " ".join(f"{np.median([r[k] for r in v]):.3f}" for k in ("largest", "cliff", "on_share", "neg_ratio", "top_input_share", "label_spread", "output_effect")))
