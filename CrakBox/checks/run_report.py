"""A run's numbers: score means against wrong items, each node's tests, and the bar. It moves no node. Usage: run_report.py RUN. Marks come from judge-calibrate."""
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

from mapper import judge as judgelib  # noqa: E402
from mapper.state import Store  # noqa: E402

run = sys.argv[1]
s = Store(os.path.join(cmd.run_dir(run), "state.db"))
b = cmd.bench(run)
cal = cmd.load_json(os.path.join(b.keys, "marks.json"))
marks = cal["marks"]
rows = {"real": [], "wrong": []}
nodes = {}
link = {"parent": [], "reader": [], "parent_wrong": [], "reader_wrong": []}
order = {"real": [], "wrong": []}
for item, k in b.key.items():
    answers = {j: a for j, a in s.answers(item).items() if ("ranked" in a or "pick" in a)}
    if not answers:
        continue
    scores = b.judge_scores(item, answers)
    if k["test"] in ("parent", "reader"):
        link[k["test"] + ("_wrong" if k["wrong_card"] else "")] += [x["correct"] for x in scores]
    elif k["test"] == "ordering":
        order["wrong" if k["wrong_card"] else "real"] += [x["rho"] for x in scores]
    else:
        rows["wrong" if k["wrong_card"] else "real"] += scores
    if not k["wrong_card"]:
        settled = judgelib.vote([judgelib.judge_passed(x, marks) for x in scores])
        entry = nodes.setdefault(k["node"], {})
        entry[k["test"]] = dict(vote=settled)
        if k["test"].startswith("pole") or k["test"].startswith("route"):
            entry[k["test"]].update({f: round(float(np.mean([x[f] for x in scores])), 2) for f in ("center_norm", "edge", "far")})
        if k["test"] == "ordering":
            entry[k["test"]]["rho"] = round(float(np.mean([x["rho"] for x in scores])), 2)


def means(rs):
    return {f: round(float(np.mean([x[f] for x in rs])), 3) for f in ("center_norm", "edge", "far", "fire_rate", "near_rate", "random_rate")} if rs else {}


status = s.status()
report = dict(marks=marks, false_pass=cal.get("false_pass"), false_pass_upper=cal.get("false_pass_upper"), wrong_items_in_marks=cal.get("wrong_items"), real=means(rows["real"]), wrong=means(rows["wrong"]), real_judgments=len(rows["real"]), wrong_judgments=len(rows["wrong"]),
              parent_right=float(np.mean(link["parent"])) if link["parent"] else None, reader_right=float(np.mean(link["reader"])) if link["reader"] else None, link_counts={k: len(v) for k, v in link.items()},
              ordering={k: round(float(np.mean(v)), 3) if v else None for k, v in order.items()}, bar=(status["understood"], status["scope"]), states=status["states"], open=status["open"])
print(json.dumps(report, indent=1))
print("\nnode, state, tests:")
for n in sorted(nodes):
    print(n, s.state(n), json.dumps(nodes[n]))
with open(os.path.join(ROOT, "checks", f"report_{run}.json"), "w") as f:
    json.dump(dict(report=report, nodes=nodes), f, indent=1)
s.close()
