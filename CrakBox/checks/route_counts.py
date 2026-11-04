"""Which workflow code would pick, on the feature survey. Triggers are read from data and not guessed."""
import collections
import json
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mapper import workflow as w
lib = w.load(os.path.join(ROOT, "workflows"))
rows = json.load(open(os.path.join(ROOT, "checks", "feature_survey.json")))
total = collections.Counter()
by = collections.defaultdict(collections.Counter)
for r in rows:
    wf = w.route(r, lib)
    total[wf.id] += 1
    by[r["node"].split(":")[0]][wf.id] += 1
print(dict(total))
for layer, c in by.items():
    print(layer, dict(c))
