"""The near-view test's result, under the marks frozen in LOG before it ran. Only each node's current card counts. Usage: arm_report.py RUN."""
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

FROZEN = dict(center=.5340473416067435, edge=.5208333333333333, far=.5)
run = sys.argv[1]
s = Store(os.path.join(cmd.run_dir(run), "state.db"))
arms = json.load(open(os.path.join(cmd.run_dir(run), "arms.json")))
old = json.load(open(os.path.join(ROOT, "checks", f"report_{run}.json")))["nodes"]
b = cmd.bench(run)
for arm in ("tool", "control"):
    votes, rows = [], []
    for node in arms[arm]:
        digest = judgelib._digest(judgelib.words_for(s.card(node), "pos", None))
        for item, k in b.key.items():
            if k["node"] == node and k["test"] == "pole:pos" and not k["wrong_card"] and k["digest"] == digest:
                answers = {j: a for j, a in s.answers(item).items() if "ranked" in a}
                if not answers:
                    continue
                scores = b.judge_scores(item, answers)
                vote = judgelib.vote([judgelib.judge_passed(x, FROZEN) for x in scores])
                votes.append(vote)
                was = old[node]["pole:pos"]
                rows.append((node, vote, {f: round(float(np.mean([x[f] for x in scores])), 2) for f in ("center_norm", "edge", "far")}, dict(was_center=was["center_norm"], was_edge=was["edge"], was_far=was["far"])))
    print(arm, "nodes", len(arms[arm]), "judged", len(votes), "passed", sum(v is True for v in votes), "failed", sum(v is False for v in votes), "split", sum(v is None for v in votes))
    print("  mean edge now", round(float(np.mean([r[2]["edge"] for r in rows])), 3) if rows else None, "was", round(float(np.mean([r[3]["was_edge"] for r in rows])), 3) if rows else None)
    for r in rows:
        print("  ", *r)
wrong = []
for item, k in b.key.items():
    if k["test"] == "pole:pos" and k["wrong_card"] and k["node"] in arms["tool"] + arms["control"]:
        answers = {j: a for j, a in s.answers(item).items() if "ranked" in a}
        if answers:
            wrong.append(judgelib.vote([judgelib.judge_passed(x, FROZEN) for x in b.judge_scores(item, answers)]))
print("wrong items under the frozen marks, all rounds on these nodes:", len(wrong), "passed", sum(v is True for v in wrong))
s.close()
