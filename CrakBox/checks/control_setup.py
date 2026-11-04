"""Step 6: 20 random neurons of ConvNeXt-Tiny's last block, seed fixed. The run is convnext_tiny. Writes the scope file."""
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = random.Random(6)
nodes = [f"c17:{j}" for j in sorted(rng.sample(range(3072), 20))]
scope = dict(nodes=nodes, goal="control: all classes", control_goal="none, this run is a control", threshold="20 random neurons, seed 6", firings="strong firings and both poles")
with open(os.path.join(ROOT, "checks", "control_scope.json"), "w") as f:
    json.dump(scope, f, indent=1)
print(" ".join(nodes))
