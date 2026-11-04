"""The negative pole, first form against words alone, on the smoke run's cards."""
import importlib.util, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("mapper_command", os.path.join(ROOT, "mapper.py"))
cmd = importlib.util.module_from_spec(spec); spec.loader.exec_module(cmd)
import numpy as np
from mapper import judge as J
from mapper.state import Store
run = "resnet18.smoke"
s = Store(os.path.join(cmd.run_dir(run), "state.db")); b = cmd.bench(run)
marks = cmd.load_json(os.path.join(b.keys, "marks.json"))["marks"]
cards = s.all_cards()
rows = {}
for item, k in b.key.items():
    if k.get("pole") != "neg" or not k["test"].startswith("pole"):
        continue
    answers = s.answers(item)
    if len(answers) < 2:
        continue
    form = "capped" if k.get("form") == 3 else ("first" if "job" in cmd.load_json(os.path.join(b.packets, item + ".json"))["words"] else "alone")
    sc = b.judge_scores(item, answers)
    rows.setdefault((form, k["wrong_card"]), []).append((k["node"], sc))
for key, v in sorted(rows.items()):
    flat = [x for _, sc in v for x in sc]
    print(key, len(v), "center", round(np.mean([x["center_norm"] for x in flat]), 3), "edge", round(np.mean([x["edge"] for x in flat]), 3), "far", round(np.mean([x["far"] for x in flat]), 3), "pass", sum(J.vote([J.passes(x, marks) for x in sc]) is True for _, sc in v), "split", sum(J.vote([J.passes(x, marks) for x in sc]) is None for _, sc in v))
for n, sc in rows.get(("capped", False), []):
    print(n, round(np.mean([x["center_norm"] for x in sc]), 2), round(np.mean([x["edge"] for x in sc]), 2), round(np.mean([x["far"] for x in sc]), 2), J.vote([J.passes(x, marks) for x in sc]))
