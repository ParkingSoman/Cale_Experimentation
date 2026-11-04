"""Step 5: pick the smoke nodes by code, so nothing is chosen for looking easy. 5 that failed the old crop test and 5 that passed. Stage 4 only, each with a main parent that has an old card, so that a parent test can be built.
Seed fixed. Prints a scope file for the run resnet18.smoke."""
import glob
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = os.path.join(ROOT, "..", "research", "clean", "curves", "map_agents", "resnet18")
scores = json.load(open(os.path.join(OLD, "review2", "new", "scores.json")))
cards = {}
for path in sorted(glob.glob(os.path.join(OLD, "cards2", "*.jsonl"))):
    for line in open(path):
        if line.strip():
            d = json.loads(line)
            cards[d["node"]] = d
real = {r["node"]: r["gap"] for r in scores if not r["wrong"] and r["node"].startswith("s4")}


def usable(n):
    parent = ((cards.get(n, {}).get("main_parent") or {}).get("node") or "").replace(" (skip)", "")
    return parent in cards


rng = random.Random(5)
failed = sorted(n for n, g in real.items() if g < .2 and usable(n))
passed = sorted(n for n, g in real.items() if g >= .2 and usable(n))
pick_f, pick_p = rng.sample(failed, 5), rng.sample(passed, 5)
print(json.dumps(dict(failed_pool=len(failed), passed_pool=len(passed), failed=pick_f, passed=pick_p), indent=1))
scope = dict(nodes=pick_f + pick_p, goal="smoke run: no goal, nodes picked to test the tool", control_goal="none, smoke run", threshold="none, smoke run", firings="strong firings and both poles")
with open(os.path.join(ROOT, "checks", "smoke_scope.json"), "w") as f:
    json.dump(scope, f, indent=1)
