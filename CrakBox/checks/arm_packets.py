"""Blind positive-pole items for the near-view test. Measures words against the judge. No node moves, and a later packet build reuses these items. Usage: arm_packets.py RUN."""
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("mapper_command", os.path.join(ROOT, "mapper.py"))
cmd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmd)
from mapper import judge as judgelib  # noqa: E402
from mapper.state import Store  # noqa: E402

run = sys.argv[1]
net = cmd.get_net(run)
s = Store(os.path.join(cmd.run_dir(run), "state.db"))
arms = json.load(open(os.path.join(cmd.run_dir(run), "arms.json")))
cards = s.all_cards()
b = cmd.bench(run)
problems, made = [], 0
for node in arms["tool"] + arms["control"]:
    if s.state(node) != "claimed":
        problems.append(f"{node}: {s.state(node)}, so it has no new card")
        continue
    try:
        before = len(b.key)
        b.build_pole(net, node, "pos", cards)
        if len(b.key) > before and b.want_wrong_item():
            b.build_pole(net, node, "pos", cards, wrong=True)
        made += len(b.key) - before
    except (judgelib.ShortPool, judgelib.Leak) as e:
        problems.append(str(e))
print(json.dumps(dict(items_made=made, problems=problems), indent=1))
s.close()
