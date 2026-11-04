"""A run's pole tests by side, real against wrong, and each judge's mean, which shows a judge who mismatched sheets. Usage: pole_report.py RUN [FORM]. Moves no node."""
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("mapper_command", os.path.join(ROOT, "mapper.py"))
cmd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmd)
import numpy as np  # noqa: E402

from mapper.state import Store  # noqa: E402

run = sys.argv[1]
form = int(sys.argv[2]) if len(sys.argv) > 2 else None
s = Store(os.path.join(cmd.run_dir(run), "state.db"))
b = cmd.bench(run)
groups, judges, pairs, per_node = {}, {}, [], {}
for item, k in b.key.items():
    if not k["test"].startswith("pole") or (form is not None and k.get("form") != form):
        continue
    answers = {j: a for j, a in s.answers(item).items() if "ranked" in a}
    if len(answers) < 2:
        continue
    scores = b.judge_scores(item, answers)
    groups.setdefault((k["pole"], k["wrong_card"]), []).extend(scores)
    for j, x in zip(answers, scores):
        judges.setdefault(j, {}).setdefault("wrong" if k["wrong_card"] else "real", []).append(x["center_norm"])
    if not k["wrong_card"]:
        pairs.append((scores[0]["center_norm"], scores[1]["center_norm"]))
        per_node.setdefault(k["node"], {})[k["pole"]] = round(float(np.mean([x["center_norm"] for x in scores])), 2)
for key in sorted(groups):
    g = groups[key]
    print(key, "judgments", len(g), {f: round(float(np.mean([x[f] for x in g])), 3) for f in ("center_norm", "edge", "far", "fire_rate", "near_rate", "random_rate")})
print("by judge, mean center:")
for j in sorted(judges):
    print(" ", j, {k: (round(float(np.mean(v)), 3), len(v)) for k, v in judges[j].items()})
a = np.array(pairs)
print("two judges on one real item, correlation of center:", round(float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), 3), "items", len(a))
print("node centers:", json.dumps(per_node))
s.close()
